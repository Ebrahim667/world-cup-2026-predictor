"""
World Cup 2026 prediction engine.

Pipeline:
  1. load_data / normalize_teams  - load raw CSVs, unify historical team names
  2. compute_elo                  - World-Football-style Elo over full history
  3. build_features               - per-match features (Elo, form, context)
  4. WCModel + train_model        - PyTorch two-headed Poisson goal model
  5. blend_fifa_into_elo          - fold current FIFA ranking into team strength
  6. simulate_tournament          - Monte Carlo of the remaining 2026 World Cup

All heavy logic lives here so it can be unit-tested from the shell and imported
by the Jupyter notebook as a thin narrative layer.
"""
from __future__ import annotations
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# Official 2026 World Cup groups (team names already match results.csv).
# ---------------------------------------------------------------------------
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Round of 32 bracket. Each slot is a tuple:
#   ("W", "A")  -> winner of group A
#   ("R", "B")  -> runner-up of group B
#   ("3", "ABCDF") -> a best-third-placed team from one of these groups
R32 = {
    73: (("R", "A"), ("R", "B")),
    74: (("W", "E"), ("3", "ABCDF")),
    75: (("W", "F"), ("R", "C")),
    76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), ("3", "CDFGH")),
    78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), ("3", "CEFHI")),
    80: (("W", "L"), ("3", "EHIJK")),
    81: (("W", "D"), ("3", "BEFIJ")),
    82: (("W", "G"), ("3", "AEHIJ")),
    83: (("R", "K"), ("R", "L")),
    84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), ("3", "EFGIJ")),
    86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), ("3", "DEIJL")),
    88: (("R", "D"), ("R", "G")),
}
# Later rounds reference winners of earlier match ids.
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL = {104: (101, 102)}

# Tournament importance weights for Elo updates / match weighting.
TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 1.00,
    "FIFA World Cup qualification": 0.75,
    "UEFA Euro": 0.85, "Copa América": 0.85, "African Cup of Nations": 0.80,
    "AFC Asian Cup": 0.75, "Gold Cup": 0.70, "UEFA Nations League": 0.75,
    "CONCACAF Nations League": 0.60, "Confederations Cup": 0.85,
    "UEFA Euro qualification": 0.60, "African Cup of Nations qualification": 0.55,
    "AFC Asian Cup qualification": 0.55, "Copa América qualification": 0.60,
    "Friendly": 0.30,
}
DEFAULT_WEIGHT = 0.50


# ---------------------------------------------------------------------------
# 1. Data loading & cleaning
# ---------------------------------------------------------------------------
def load_data():
    results = pd.read_csv(DATA / "raw/results.csv", parse_dates=["date"])
    shootouts = pd.read_csv(DATA / "raw/shootouts.csv", parse_dates=["date"])
    former = pd.read_csv(DATA / "raw/former_names.csv")
    return results, shootouts, former


def normalize_teams(results, former):
    """Map historical / former country names onto their modern equivalent."""
    mapping = dict(zip(former["former"], former["current"]))
    # A few names that appear in results.csv but not in former_names.csv.
    mapping.update({
        "Czechia": "Czech Republic", "Türkiye": "Turkey",
        "Zaire": "DR Congo", "Congo DR": "DR Congo",
    })
    results = results.copy()
    for col in ("home_team", "away_team"):
        results[col] = results[col].replace(mapping)
    return results


def tournament_weight(name):
    return TOURNAMENT_WEIGHT.get(name, DEFAULT_WEIGHT)


# ---------------------------------------------------------------------------
# 2. Elo ratings (World Football Elo style)
# ---------------------------------------------------------------------------
def _goal_multiplier(goal_diff):
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0


def compute_elo(results, base=1500.0, K=40.0, hfa=65.0):
    """Return (elo_pre_df, final_ratings).

    elo_pre_df holds each match's pre-match ratings (features); final_ratings is
    the dict of ratings after the last *played* match. Unplayed (NaN-score) rows
    receive pre-match ratings but never update the table.
    """
    ratings: dict[str, float] = {}
    pre_home, pre_away = {}, {}
    order = results.sort_values("date").index
    for idx in order:
        row = results.loc[idx]
        h, a = row.home_team, row.away_team
        rh = ratings.get(h, base)
        ra = ratings.get(a, base)
        pre_home[idx], pre_away[idx] = rh, ra
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue
        ha = 0.0 if bool(row.neutral) else hfa
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + ha - ra) / 400.0)))
        gd = row.home_score - row.away_score
        s = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        delta = K * tournament_weight(row.tournament) * _goal_multiplier(gd) * (s - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    elo_pre = pd.DataFrame({"home_elo_pre": pre_home, "away_elo_pre": pre_away})
    return elo_pre, ratings


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
FEATURES = [
    "elo_diff", "neutral", "importance",
    "home_form_gf", "home_form_ga", "home_form_pts",
    "away_form_gf", "away_form_ga", "away_form_pts",
]


def _rolling_form(results, n=10):
    """Pre-match rolling means of goals-for / against / points per team."""
    home = results[["date", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "team", "gf", "ga"]
    home["idx"], home["side"] = results.index, "home"
    away = results[["date", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "team", "gf", "ga"]
    away["idx"], away["side"] = results.index, "away"
    long = pd.concat([home, away]).sort_values(["team", "date"])
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga],
                            [3.0, 1.0], default=0.0)
    for col in ("gf", "ga", "pts"):
        long[f"form_{col}"] = (long.groupby("team")[col]
                               .transform(lambda s: s.shift().rolling(n, min_periods=1).mean()))
    return long


def build_features(results, elo_pre, hfa=65.0):
    df = results.join(elo_pre)
    df["neutral"] = df["neutral"].astype(float)
    df["importance"] = df["tournament"].map(tournament_weight).fillna(DEFAULT_WEIGHT)
    df["elo_diff"] = (df["home_elo_pre"] + (1 - df["neutral"]) * hfa) - df["away_elo_pre"]

    long = _rolling_form(results)
    h = long[long.side == "home"].set_index("idx")[["form_gf", "form_ga", "form_pts"]]
    a = long[long.side == "away"].set_index("idx")[["form_gf", "form_ga", "form_pts"]]
    df["home_form_gf"], df["home_form_ga"], df["home_form_pts"] = h.form_gf, h.form_ga, h.form_pts
    df["away_form_gf"], df["away_form_ga"], df["away_form_pts"] = a.form_gf, a.form_ga, a.form_pts

    # Latest known form per team -> used at prediction / simulation time.
    latest = (long.sort_values("date").groupby("team")
              .agg(form_gf=("form_gf", "last"), form_ga=("form_ga", "last"),
                   form_pts=("form_pts", "last")))
    latest_form = latest.fillna(latest.mean()).to_dict("index")
    return df, latest_form


# ---------------------------------------------------------------------------
# 4. PyTorch Poisson goal model
# ---------------------------------------------------------------------------
class WCModel(nn.Module):
    """Shared trunk -> two heads producing log expected goals (home, away)."""

    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head_home = nn.Linear(hidden, 1)
        self.head_away = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.trunk(x)
        log_lh = self.head_home(z).squeeze(-1)
        log_la = self.head_away(z).squeeze(-1)
        return log_lh, log_la


def _sample_weights(df, half_life_years=8.0):
    age = (df["date"].max() - df["date"]).dt.days / 365.25
    recency = 0.5 ** (age / half_life_years)
    return (recency * df["importance"]).to_numpy()


def train_model(df, epochs=60, lr=1e-3, batch=512, val_frac=0.1, seed=0,
                verbose=True):
    """Train on played matches; chronological train/val split. Returns
    (model, scaler_dict, history)."""
    torch.manual_seed(seed); np.random.seed(seed)
    played = df.dropna(subset=["home_score", "away_score", "elo_diff",
                               "home_form_gf", "away_form_gf"]).sort_values("date")
    X = played[FEATURES].to_numpy(np.float32)
    yh = played["home_score"].to_numpy(np.float32)
    ya = played["away_score"].to_numpy(np.float32)
    w = _sample_weights(played).astype(np.float32)

    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    scaler = {"mu": mu, "sd": sd}

    n_val = int(len(Xs) * val_frac)
    tr = slice(0, len(Xs) - n_val); va = slice(len(Xs) - n_val, len(Xs))
    to_t = lambda arr: torch.tensor(arr)
    Xtr, Xva = to_t(Xs[tr]), to_t(Xs[va])
    yhtr, yatr = to_t(yh[tr]), to_t(ya[tr])
    yhva, yava = to_t(yh[va]), to_t(ya[va])
    wtr = to_t(w[tr])

    model = WCModel(len(FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pois = nn.PoissonNLLLoss(log_input=True, reduction="none")
    history = []
    n = Xtr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            lh, la = model(Xtr[b])
            loss = ((pois(lh, yhtr[b]) + pois(la, yatr[b])) * wtr[b]).mean()
            loss.backward(); opt.step()
        if verbose or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                lh, la = model(Xva)
                vloss = (pois(lh, yhva) + pois(la, yava)).mean().item()
                # outcome accuracy on val
                ph, pa = torch.exp(lh).numpy(), torch.exp(la).numpy()
                pred = np.sign(ph - pa); true = np.sign(yhva.numpy() - yava.numpy())
                acc = (pred == true).mean()
            history.append({"epoch": ep, "val_loss": vloss, "val_outcome_acc": acc})
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"epoch {ep:3d}  val_poisson_nll={vloss:.4f}  outcome_acc={acc:.3f}")
    return model, scaler, history


# ---------------------------------------------------------------------------
# 5. Squad strength (player ratings) + external-strength blend
# ---------------------------------------------------------------------------
# Map our team names -> nationality strings used in the EA FC 26 player dataset.
SQUAD_NAME_MAP = {
    "South Korea": "Korea Republic", "Ivory Coast": "Côte d'Ivoire",
    "DR Congo": "Congo DR", "Turkey": "Türkiye", "Czech Republic": "Czechia",
    "Cape Verde": "Cabo Verde", "Curaçao": "Curacao",
}
SQUAD_SIZE = 23
REPLACEMENT_RATING = 62.0  # rating assigned to "missing" depth slots


def build_squad_strength(players_csv=DATA / "external/eafc26_players_raw.csv",
                         out_csv=DATA / "external/squad_strength_2026.csv"):
    """Compute a per-team squad-strength index from individual player ratings.

    For each WC team we take its top-rated players (by EA `overall`), regularise
    for *depth* (a squad with fewer than 23 rated players has the missing slots
    filled at replacement level), and record star power and squad value too.
    Changing the player CSV (new players, retirements, different call-ups)
    directly changes these numbers -> changes the predictions.
    """
    p = pd.read_csv(players_csv, low_memory=False)
    p = p.sort_values("overall", ascending=False).drop_duplicates("player_id")
    teams = [t for g in GROUPS.values() for t in g]
    rows = []
    for t in teams:
        nat = SQUAD_NAME_MAP.get(t, t)
        sq = p[p.nationality_name == nat].sort_values("overall", ascending=False).head(SQUAD_SIZE)
        ov = sq.overall.to_numpy(float)
        n = len(ov)
        # depth-regularised mean: pad missing slots with replacement level
        depth_overall = (ov.sum() + (SQUAD_SIZE - n) * REPLACEMENT_RATING) / SQUAD_SIZE
        rows.append({
            "team": t, "n_players": n,
            "squad_overall": round(depth_overall, 2),
            "raw_overall": round(ov.mean(), 2) if n else np.nan,
            "top3_overall": round(np.sort(ov)[::-1][:3].mean(), 2) if n else np.nan,
            "squad_value_m": round(sq.value_eur.sum() / 1e6, 1),
        })
    sqdf = pd.DataFrame(rows).sort_values("squad_overall", ascending=False).reset_index(drop=True)
    sqdf.to_csv(out_csv, index=False)
    return sqdf


def blend_external_into_elo(ratings, alpha_fifa=0.20, alpha_squad=0.20):
    """Nudge each WC team's Elo toward its current FIFA ranking AND its squad
    (player-rating) strength. All three signals are standardised within the 48
    teams, blended in z-space, then mapped back to the Elo scale.

    alpha_fifa + alpha_squad is the total weight given to "current" info; the
    remainder stays with the results-based historical Elo.
    """
    teams = [t for g in GROUPS.values() for t in g]
    elo = np.array([ratings.get(t, 1500.0) for t in teams])

    fifa = pd.read_csv(DATA / "external/fifa_rankings_2026.csv")
    fmap = dict(zip(fifa.team, fifa.fifa_points))
    pts = np.array([fmap.get(t, np.nan) for t in teams])
    pts = np.where(np.isnan(pts), np.nanmean(pts), pts)

    sq_path = DATA / "external/squad_strength_2026.csv"
    if not sq_path.exists():
        build_squad_strength()
    sq = pd.read_csv(sq_path)
    smap = dict(zip(sq.team, sq.squad_overall))
    squad = np.array([smap.get(t, np.nan) for t in teams])
    squad = np.where(np.isnan(squad), np.nanmean(squad), squad)

    z = lambda v: (v - v.mean()) / v.std()
    alpha_elo = 1.0 - alpha_fifa - alpha_squad
    z_blend = alpha_elo * z(elo) + alpha_fifa * z(pts) + alpha_squad * z(squad)
    blended = elo.mean() + z_blend * elo.std()

    out = dict(ratings)
    for t, v in zip(teams, blended):
        out[t] = float(v)
    return out


def blend_fifa_into_elo(ratings, alpha=0.30):
    """Backwards-compatible FIFA-only blend (squad weight = 0)."""
    return blend_external_into_elo(ratings, alpha_fifa=alpha, alpha_squad=0.0)


# ---------------------------------------------------------------------------
# 6. Match prediction
# ---------------------------------------------------------------------------
def predict_lambdas(model, scaler, ratings, latest_form, home, away,
                    neutral=True, hfa=65.0):
    """Expected goals (lambda_home, lambda_away) for a single fixture."""
    rh, ra = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
    elo_diff = rh + (0 if neutral else hfa) - ra
    fh = latest_form.get(home, {"form_gf": 1.2, "form_ga": 1.2, "form_pts": 1.3})
    fa = latest_form.get(away, {"form_gf": 1.2, "form_ga": 1.2, "form_pts": 1.3})
    x = np.array([[elo_diff, float(neutral), TOURNAMENT_WEIGHT["FIFA World Cup"],
                   fh["form_gf"], fh["form_ga"], fh["form_pts"],
                   fa["form_gf"], fa["form_ga"], fa["form_pts"]]], np.float32)
    xs = (x - scaler["mu"]) / scaler["sd"]
    model.eval()
    with torch.no_grad():
        lh, la = model(torch.tensor(xs))
    return float(np.exp(lh.item())), float(np.exp(la.item()))


def outcome_probs(model, scaler, ratings, latest_form, home, away,
                  neutral=True, max_goals=10):
    """Win/draw/loss probabilities via independent Poisson goal grid."""
    lh, la = predict_lambdas(model, scaler, ratings, latest_form, home, away, neutral)
    gh = np.exp(-lh) * lh ** np.arange(max_goals + 1) / np.array(
        [math.factorial(k) for k in range(max_goals + 1)])
    ga = np.exp(-la) * la ** np.arange(max_goals + 1) / np.array(
        [math.factorial(k) for k in range(max_goals + 1)])
    grid = np.outer(gh, ga)
    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()
    return {"lambda_home": lh, "lambda_away": la,
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away}


# ---------------------------------------------------------------------------
# 7. Monte Carlo tournament simulation
# ---------------------------------------------------------------------------
def _played_group_results(results):
    """Real 2026 World Cup group scores already in the dataset."""
    wc = results[(results.date.dt.year == 2026) &
                 (results.tournament == "FIFA World Cup") &
                 results.home_score.notna()]
    return [(r.home_team, r.away_team, int(r.home_score), int(r.away_score))
            for r in wc.itertuples()]


def _assign_thirds(third_teams_by_group, slots, rng):
    """Backtracking match of advancing third-placed teams to bracket slots."""
    groups_avail = list(third_teams_by_group.keys())
    order = sorted(range(len(slots)), key=lambda i: len(slots[i]))  # tightest first
    assignment = {}

    def bt(k, used):
        if k == len(order):
            return True
        si = order[k]
        allowed = [g for g in slots[si] if g in groups_avail and g not in used]
        rng.shuffle(allowed)
        for g in allowed:
            assignment[si] = g
            if bt(k + 1, used | {g}):
                return True
        return False

    if not bt(0, set()):  # fallback: ignore constraints
        free = list(groups_avail)
        rng.shuffle(free)
        for i in range(len(slots)):
            assignment[i] = free[i]
    return {i: third_teams_by_group[assignment[i]] for i in range(len(slots))}


def _knockout(home, away, model, scaler, ratings, latest_form, rng):
    lh, la = predict_lambdas(model, scaler, ratings, latest_form, home, away, neutral=True)
    gh, ga = rng.poisson(lh), rng.poisson(la)
    if gh > ga:
        return home
    if ga > gh:
        return away
    # shootout: slight edge to the stronger side
    rh, ra = ratings.get(home, 1500), ratings.get(away, 1500)
    p = 1.0 / (1.0 + 10 ** (-((rh - ra) / 400.0)))
    return home if rng.random() < p else away


def simulate_once(model, scaler, ratings, latest_form, played, rng):
    """One full tournament. Returns dict: stage reached per team + champion."""
    reached = {t: "Group" for g in GROUPS.values() for t in g}
    played_map = {(h, a): (hs, as_) for h, a, hs, as_ in played}

    # ----- group stage -----
    standings = {}
    third_place = []
    for gl, teams in GROUPS.items():
        tab = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
        for h, a in itertools.combinations(teams, 2):
            if (h, a) in played_map:
                hs, as_ = played_map[(h, a)]
            elif (a, h) in played_map:
                as_, hs = played_map[(a, h)]
            else:
                lh, la = predict_lambdas(model, scaler, ratings, latest_form, h, a)
                hs, as_ = rng.poisson(lh), rng.poisson(la)
            tab[h]["gf"] += hs; tab[h]["ga"] += as_
            tab[a]["gf"] += as_; tab[a]["ga"] += hs
            if hs > as_:
                tab[h]["pts"] += 3
            elif as_ > hs:
                tab[a]["pts"] += 3
            else:
                tab[h]["pts"] += 1; tab[a]["pts"] += 1
        ranked = sorted(teams, key=lambda t: (tab[t]["pts"],
                        tab[t]["gf"] - tab[t]["ga"], tab[t]["gf"], rng.random()),
                        reverse=True)
        standings[gl] = ranked
        third_place.append((gl, ranked[2],
                            tab[ranked[2]]["pts"],
                            tab[ranked[2]]["gf"] - tab[ranked[2]]["ga"],
                            tab[ranked[2]]["gf"]))

    # best 8 thirds
    third_place.sort(key=lambda x: (x[2], x[3], x[4], rng.random()), reverse=True)
    best_thirds = {gl: team for gl, team, *_ in third_place[:8]}
    for gl, ranked in standings.items():
        for t in ranked[:2]:
            reached[t] = "R32"
    for t in best_thirds.values():
        reached[t] = "R32"

    # ----- assign third slots -----
    third_slot_ids = [mid for mid, (s1, s2) in R32.items() if s2[0] == "3"]
    slot_constraints = [set(R32[mid][1][1]) for mid in third_slot_ids]
    assigned = _assign_thirds(best_thirds, slot_constraints, rng)
    third_for_match = {mid: assigned[i] for i, mid in enumerate(third_slot_ids)}

    def resolve(slot, mid):
        kind, val = slot
        if kind == "W":
            return standings[val][0]
        if kind == "R":
            return standings[val][1]
        return third_for_match[mid]

    # ----- knockout rounds -----
    winners = {}
    for mid, (s1, s2) in R32.items():
        t1, t2 = resolve(s1, mid), resolve(s2, mid)
        winners[mid] = _knockout(t1, t2, model, scaler, ratings, latest_form, rng)
    for t in winners.values():
        reached[t] = "R16"

    def play_round(round_def, stage):
        for mid, (m1, m2) in round_def.items():
            winners[mid] = _knockout(winners[m1], winners[m2], model, scaler,
                                     ratings, latest_form, rng)
            reached[winners[mid]] = stage

    play_round(R16, "QF")
    play_round(QF, "SF")
    play_round(SF, "Final")
    champion = _knockout(winners[101], winners[102], model, scaler,
                         ratings, latest_form, rng)
    reached[champion] = "Champion"
    return reached, champion


STAGES = ["Group", "R32", "R16", "QF", "SF", "Final", "Champion"]
STAGE_RANK = {s: i for i, s in enumerate(STAGES)}


def simulate_tournament(model, scaler, ratings, latest_form, played,
                        n_sims=10000, seed=0):
    rng = np.random.default_rng(seed)
    teams = [t for g in GROUPS.values() for t in g]
    counts = {t: {s: 0 for s in STAGES} for t in teams}
    titles = {t: 0 for t in teams}
    for _ in range(n_sims):
        reached, champ = simulate_once(model, scaler, ratings, latest_form, played, rng)
        for t, st in reached.items():
            # count team as having *reached at least* each stage up to st
            for s in STAGES[:STAGE_RANK[st] + 1]:
                counts[t][s] += 1
        titles[champ] += 1
    rows = []
    for t in teams:
        rows.append({"team": t,
                     "win_pct": 100 * titles[t] / n_sims,
                     "final_pct": 100 * counts[t]["Final"] / n_sims,
                     "semi_pct": 100 * counts[t]["SF"] / n_sims,
                     "quarter_pct": 100 * counts[t]["QF"] / n_sims,
                     "r16_pct": 100 * counts[t]["R16"] / n_sims,
                     "advance_pct": 100 * counts[t]["R32"] / n_sims})
    return pd.DataFrame(rows).sort_values("win_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: full pipeline (used by the notebook and the self-test).
# ---------------------------------------------------------------------------
def run_pipeline(n_sims=2000, epochs=60, verbose=True,
                 alpha_fifa=0.20, alpha_squad=0.20):
    results, shootouts, former = load_data()
    results = normalize_teams(results, former)
    elo_pre, ratings = compute_elo(results)
    df, latest_form = build_features(results, elo_pre)
    model, scaler, history = train_model(df, epochs=epochs, verbose=verbose)
    build_squad_strength()
    ratings_b = blend_external_into_elo(ratings, alpha_fifa=alpha_fifa,
                                        alpha_squad=alpha_squad)
    played = _played_group_results(results)
    sim = simulate_tournament(model, scaler, ratings_b, latest_form, played,
                              n_sims=n_sims)
    return dict(results=results, df=df, ratings=ratings, ratings_b=ratings_b,
                latest_form=latest_form, model=model, scaler=scaler,
                history=history, played=played, sim=sim)


if __name__ == "__main__":
    out = run_pipeline(n_sims=1000, epochs=40, verbose=True)
    print("\nPlayed group matches loaded:", len(out["played"]))
    print("\nTop 12 title probabilities:")
    print(out["sim"].head(12).to_string(index=False))

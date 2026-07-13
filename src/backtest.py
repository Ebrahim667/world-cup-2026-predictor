"""
Backtest: would this pipeline have predicted past World Cup winners?

For each past tournament we:
  1. Cut the match history to matches *strictly before* the tournament kicked off
     (no leakage: the tournament's own games are never seen in training).
  2. Rebuild Elo and retrain the Poisson NN on that cut history only.
  3. Simulate the real 32-team bracket for that year (Monte Carlo).
  4. Report the title-odds board and where the actual winner ranked.

Note: we deliberately use *pure historical Elo* here (no FIFA/squad blend) —
the only FIFA-ranking and EA-player data we have is from 2026, so using it for
2018/2022 would be future leakage. This is the honest, time-aware setup.

Run:  PYTHONIOENCODING=utf-8 python src/backtest.py
"""
from __future__ import annotations
import itertools
import numpy as np

import worldcup as w

# ---------------------------------------------------------------------------
# Past tournaments: real groups + standard 32-team bracket template.
# Team names match results.csv after normalize_teams().
# ---------------------------------------------------------------------------
WC2018 = {
    "start": "2018-06-14",
    "winner": "France",
    "groups": {
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"],
    },
}

WC2022 = {
    "start": "2022-11-20",
    "winner": "Argentina",
    "groups": {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    },
}

# Standard 32-team bracket template. Each R16 slot references a group position:
#   ("W","A") = winner of group A, ("R","A") = runner-up of group A.
# Verified against the real 2018 (France path) and 2022 (Argentina path) draws.
R16_T = {
    1: (("W", "A"), ("R", "B")),
    2: (("W", "C"), ("R", "D")),
    3: (("W", "E"), ("R", "F")),
    4: (("W", "G"), ("R", "H")),
    5: (("W", "B"), ("R", "A")),
    6: (("W", "D"), ("R", "C")),
    7: (("W", "F"), ("R", "E")),
    8: (("W", "H"), ("R", "G")),
}
QF_T = {11: (1, 2), 12: (3, 4), 13: (5, 6), 14: (7, 8)}
SF_T = {21: (11, 12), 22: (13, 14)}
FINAL_T = (21, 22)


def _group_standings(groups, model, scaler, ratings, latest_form, rng):
    standings = {}
    for gl, teams in groups.items():
        tab = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
        for h, a in itertools.combinations(teams, 2):
            lh, la = w.predict_lambdas(model, scaler, ratings, latest_form, h, a)
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
    return standings


def _simulate_once(groups, model, scaler, ratings, latest_form, rng):
    standings = _group_standings(groups, model, scaler, ratings, latest_form, rng)

    def resolve(slot):
        kind, g = slot
        return standings[g][0] if kind == "W" else standings[g][1]

    winners = {}
    for mid, (s1, s2) in R16_T.items():
        winners[mid] = w._knockout(resolve(s1), resolve(s2), model, scaler,
                                   ratings, latest_form, rng)
    for rnd in (QF_T, SF_T):
        for mid, (m1, m2) in rnd.items():
            winners[mid] = w._knockout(winners[m1], winners[m2], model, scaler,
                                       ratings, latest_form, rng)
    champ = w._knockout(winners[FINAL_T[0]], winners[FINAL_T[1]], model, scaler,
                        ratings, latest_form, rng)
    return champ


def backtest(tourney, n_sims=5000, epochs=40, seed=0):
    name = tourney["start"][:4]
    print(f"\n{'='*60}\n  Backtesting World Cup {name}  (actual winner: {tourney['winner']})\n{'='*60}")
    results, shootouts, former = w.load_data()
    results = w.normalize_teams(results, former)

    # ---- leakage guard: keep only matches before the tournament started ----
    cut = results[results["date"] < tourney["start"]].copy()
    print(f"Training on {len(cut):,} matches up to {tourney['start']} "
          f"(full history is {len(results):,}).")

    elo_pre, ratings = w.compute_elo(cut)
    df, latest_form = w.build_features(cut, elo_pre)
    model, scaler, _ = w.train_model(df, epochs=epochs, verbose=False)

    rng = np.random.default_rng(seed)
    teams = [t for g in tourney["groups"].values() for t in g]
    titles = {t: 0 for t in teams}
    for _ in range(n_sims):
        titles[_simulate_once(tourney["groups"], model, scaler, ratings, latest_form, rng)] += 1

    board = sorted(teams, key=lambda t: titles[t], reverse=True)
    print(f"\nPre-tournament title odds (top 10):")
    for i, t in enumerate(board[:10], 1):
        star = "  <-- actual winner" if t == tourney["winner"] else ""
        print(f"  {i:2d}. {t:<16} {100*titles[t]/n_sims:5.1f}%{star}")

    rank = board.index(tourney["winner"]) + 1
    print(f"\n>> Model ranked the actual winner ({tourney['winner']}) #{rank} of {len(teams)} "
          f"at {100*titles[tourney['winner']]/n_sims:.1f}% title odds.")
    return rank, 100 * titles[tourney["winner"]] / n_sims


if __name__ == "__main__":
    backtest(WC2018)
    backtest(WC2022)

"""Builds notebooks/world_cup_2026_prediction.ipynb from tested src/worldcup.py."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 🏆 World Cup 2026 — Match & Champion Predictor

A neural-network pipeline that predicts every remaining match of the 2026 FIFA
World Cup and simulates the tournament to a champion.

**Approach**
1. **Data** — 49k international matches (1872–2026), goalscorers, shootouts.
2. **Elo ratings** built from the full match history (team strength).
3. **Features** — Elo difference, recent form, neutral venue, match importance.
4. **Model** — a PyTorch two-headed *Poisson* network predicting expected goals
   for each side (recency- and importance-weighted training).
5. **FIFA rankings (June 2026)** and **squad strength from individual player
   ratings (EA FC 26)** blended into pre-tournament strength.
6. **Monte-Carlo simulation** of the remaining group + knockout matches → title
   odds, compared against **bookmaker odds**.

> **Players:** team strength is adjusted by the quality of each squad's actual
> players (top-23 EA ratings, depth-regularised). Change the player data and the
> predictions change — see §6.

All heavy logic lives in `src/worldcup.py`; this notebook is the narrative.""")

code("""import sys, math
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))   # run from notebooks/
sys.path.insert(0, "src")                              # or from project root
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import worldcup as wc
pd.set_option("display.max_rows", 60)
print("Engine loaded. Torch:", __import__("torch").__version__)""")

md("## 1. Load data & quality check")
code("""results, shootouts, former = wc.load_data()
print("results :", results.shape)
print("shootouts:", shootouts.shape)
print("former names:", former.shape)
print("\\nMissing scores:", results[['home_score','away_score']].isna().any(axis=1).sum(),
      "(all are unplayed 2026 WC fixtures)")
print("Date range:", results.date.min().date(), "->", results.date.max().date())
print("Duplicate rows:", results.duplicated().sum())

wc2026 = results[(results.date.dt.year==2026) & (results.tournament=='FIFA World Cup')]
print("\\n2026 WC fixtures:", len(wc2026),
      "| already played:", wc2026.home_score.notna().sum(),
      "| remaining:", wc2026.home_score.isna().sum())
wc2026[wc2026.home_score.notna()][['date','home_team','away_team','home_score','away_score']]""")

md("""## 2. Normalize team names

Historical names (e.g. *West Germany → Germany*, *Zaire → DR Congo*) are mapped
to their modern equivalents so a country's strength is continuous through time.""")
code("""results = wc.normalize_teams(results, former)
former.head()""")

md("""## 3. Elo ratings from full history

A World-Football-style Elo: K-factor scaled by **match importance** (World Cup
counts more than friendlies) and **goal margin**. Home advantage is applied
unless the match is on neutral ground. We store each match's *pre-match* ratings
(model features) and the *final* ratings (simulation starting strength).""")
code("""elo_pre, ratings = wc.compute_elo(results)
top = (pd.Series(ratings).sort_values(ascending=False).head(20)
       .rename("elo").round(0).reset_index().rename(columns={'index':'team'}))
top.index += 1
top""")

md("""## 4. Feature engineering

Per match we build: `elo_diff` (incl. home advantage), `neutral`, `importance`,
and pre-match **rolling form** (last 10 matches' goals for / against / points)
for each side. We also cache each team's *latest* form for prediction time.""")
code("""df, latest_form = wc.build_features(results, elo_pre)
print("Feature columns:", wc.FEATURES)
df.dropna(subset=wc.FEATURES)[['date','home_team','away_team','home_score',
    'away_score'] + wc.FEATURES].tail(6)""")

md("""## 5. Train the neural network

A shared MLP trunk with two heads outputs **log expected goals** for home and
away. Trained with **Poisson negative-log-likelihood**, weighted so recent and
important matches dominate. Chronological train/validation split.""")
code("""model, scaler, history = wc.train_model(df, epochs=60, verbose=True)
hist = pd.DataFrame(history)""")

code("""fig, ax = plt.subplots(1, 2, figsize=(11,3.5))
ax[0].plot(hist.epoch, hist.val_loss); ax[0].set_title("Validation Poisson NLL")
ax[0].set_xlabel("epoch")
ax[1].plot(hist.epoch, hist.val_outcome_acc, color="green")
ax[1].set_title("Validation outcome accuracy (W/D/L)"); ax[1].set_xlabel("epoch")
plt.tight_layout(); plt.show()
print("Final val outcome accuracy: %.3f" % hist.val_outcome_acc.iloc[-1])
print("(A naive 'home/stronger team always wins' baseline sits well below this.)")""")

md("""## 6. Player ratings → squad strength, then blend into team strength

This is where **individual players matter**. From the EA FC 26 player database
(18k players) we take each nation's **top-23 players by `overall` rating**,
regularise for **depth** (thin squads get replacement-level slots), and form a
`squad_overall` index per team. Star power (`top3_overall`) and total squad
`value` are recorded too.

If you swap the player file — a new wonderkid appears, a star retires, a
different 23 are called up — these numbers move, and so do the predictions.

Neither FIFA points nor squad ratings can be *training* features (we have no
historical per-match values), so we use them the correct way: nudge each team's
results-based Elo toward its **current FIFA ranking** and **current squad
strength** (`alpha_fifa=0.20`, `alpha_squad=0.20`; 0.60 stays with history).""")
code("""squad = wc.build_squad_strength()
print("Squad-strength index (top-23 EA ratings, depth-regularised):")
squad.head(15).reset_index(drop=True)""")
code("""# A few player squads behind the numbers
players = pd.read_csv(wc.DATA / "external/eafc26_players_raw.csv", low_memory=False)
players = players.sort_values("overall", ascending=False).drop_duplicates("player_id")
def squad_of(team, k=8):
    nat = wc.SQUAD_NAME_MAP.get(team, team)
    s = players[players.nationality_name == nat].nlargest(k, "overall")
    return s[["short_name","player_positions","overall","age","club_name"]].reset_index(drop=True)
print("Top players — France:"); display(squad_of("France"))
print("Top players — Argentina:"); display(squad_of("Argentina"))""")
code("""ratings_b = wc.blend_external_into_elo(ratings, alpha_fifa=0.20, alpha_squad=0.20)
fifa = pd.read_csv(wc.DATA / "external/fifa_rankings_2026.csv")
teams = [t for g in wc.GROUPS.values() for t in g]
comp = pd.DataFrame({"team": teams,
                     "elo_history": [round(ratings[t]) for t in teams],
                     "elo_blended": [round(ratings_b[t]) for t in teams]})
comp = comp.merge(fifa[['team','fifa_rank']], on='team', how='left')
comp = comp.merge(squad[['team','squad_overall']], on='team', how='left')
comp.sort_values("elo_blended", ascending=False).head(15).reset_index(drop=True)""")

md("""## 7. Predict any single match

`outcome_probs` returns expected goals and Win/Draw/Loss probabilities from an
independent-Poisson goal grid. Try your own fixtures here.""")
code("""def show_match(h, a, neutral=True):
    r = wc.outcome_probs(model, scaler, ratings_b, latest_form, h, a, neutral)
    print(f"{h} vs {a}  (neutral={neutral})")
    print(f"  expected score:  {h} {r['lambda_home']:.2f} - {r['lambda_away']:.2f} {a}")
    print(f"  P({h} win)={r['p_home']:.1%}  P(draw)={r['p_draw']:.1%}  P({a} win)={r['p_away']:.1%}\\n")

show_match("Spain", "France")
show_match("Argentina", "Brazil")
show_match("United States", "England")""")

md("""## 8. Predict every remaining group match

Expected score and W/D/L probabilities for all not-yet-played group fixtures.
Saved to `reports/remaining_group_predictions.csv`.""")
code("""rem = wc2026[wc2026.home_score.isna()].copy()
rows = []
for r in rem.itertuples():
    o = wc.outcome_probs(model, scaler, ratings_b, latest_form, r.home_team, r.away_team)
    rows.append({"date": r.date.date(), "home": r.home_team, "away": r.away_team,
                 "xG_home": round(o['lambda_home'],2), "xG_away": round(o['lambda_away'],2),
                 "P_home": round(o['p_home'],3), "P_draw": round(o['p_draw'],3),
                 "P_away": round(o['p_away'],3)})
pred = pd.DataFrame(rows)
pred.to_csv(wc.ROOT / "reports/remaining_group_predictions.csv", index=False)
print("Saved", len(pred), "match predictions to reports/remaining_group_predictions.csv")
pred""")

md("""## 9. Monte-Carlo tournament simulation

Simulate the remaining tournament 10,000× — sampling goals from the model's
Poisson rates, applying real group results already played, group tiebreakers,
the official 48-team bracket (12 winners + 12 runners-up + 8 best thirds), and
shootouts for knockout draws.""")
code("""played = wc._played_group_results(results)
print("Real group results fed in:", len(played))
sim = wc.simulate_tournament(model, scaler, ratings_b, latest_form, played,
                             n_sims=10000, seed=42)
sim.to_csv(wc.ROOT / "reports/title_probabilities.csv", index=False)
sim.head(20).reset_index(drop=True)""")

code("""top12 = sim.head(12)
plt.figure(figsize=(9,4.5))
plt.barh(top12.team[::-1], top12.win_pct[::-1], color="#d4af37")
plt.xlabel("Title probability (%)"); plt.title("World Cup 2026 — title odds (model)")
for i,(t,v) in enumerate(zip(top12.team[::-1], top12.win_pct[::-1])):
    plt.text(v+0.2, i, f"{v:.1f}%", va="center")
plt.tight_layout(); plt.savefig(wc.ROOT/"reports/title_odds.png", dpi=120); plt.show()""")

md("""## 10. Sanity check vs the bookmakers

Our model's title probabilities next to bookmaker / prediction-market implied
probabilities (June 12 2026). Differences are expected — bookmakers price in
squad news and sentiment; our model is strength-driven.""")
code("""odds = pd.read_csv(wc.DATA / "external/bookmaker_odds_2026.csv")
cmp = sim[['team','win_pct']].merge(
        odds[['team','implied_prob_pct']].rename(columns={'implied_prob_pct':'market_pct'}),
        on='team', how='right')
cmp['model_pct'] = cmp['win_pct'].round(1)
cmp[['team','model_pct','market_pct']].sort_values('market_pct', ascending=False).reset_index(drop=True)""")

md("""## 11. Most likely champion & bracket depth

Probability each top team reaches each stage.""")
code("""stage_cols = ['advance_pct','r16_pct','quarter_pct','semi_pct','final_pct','win_pct']
view = sim.head(12).set_index('team')[stage_cols]
view.columns = ['Reach R32','Reach R16','Reach QF','Reach SF','Reach Final','Win']
view.round(1)""")

code("""champ = sim.iloc[0]
print("="*48)
print(f"  MODEL FAVOURITE TO WIN 2026 WORLD CUP")
print(f"  >>> {champ.team}  ({champ.win_pct:.1f}% title probability) <<<")
print("="*48)
print("\\nReports written to the reports/ folder:")
for f in ['title_probabilities.csv','remaining_group_predictions.csv','title_odds.png']:
    print("  -", f)""")

md("""## 12. How to re-run / extend

* **More data each matchday** — as results come in, the `results.csv` is updated;
  just re-run the notebook. Already-played group games are fed in as fixed results
  and the simulation only randomises what's left.
* **Players** — `data/external/eafc26_players_raw.csv` drives squad strength;
  replace it (or edit a team's players) and re-run §6 to see predictions change.
* **Knobs** — `alpha_fifa` / `alpha_squad` (blend weights), Elo `K`/`hfa` in
  `compute_elo`, `epochs`, and `n_sims` all live in `src/worldcup.py`.
* **One-shot** — `wc.run_pipeline(n_sims=10000)` runs the whole thing headless.

> Predictions are probabilistic: even the favourite wins only ~1 time in 4. Treat
> them as odds, not certainties. ⚽""")

nb["cells"] = cells
out = ROOT / "notebooks" / "world_cup_2026_prediction.ipynb"
nbf.write(nb, str(out))
print("Wrote", out)

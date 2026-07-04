import sys, os, json, bisect, warnings
from datetime import datetime
from io import StringIO
import numpy as np
import polars as pl
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

sys.argv = ["debug", "--max-date", "2026-06-27"]

import stacking_model as sm

sm.MAX_DATE = "2026-06-27"

static_elo = sm.load_static_elo()
elo_lookup = sm.load_elo_history()

r = requests.get(sm.RESULTS_URL, timeout=30)
r.raise_for_status()
df = pl.read_csv(StringIO(r.text), schema_overrides={"home_score": pl.Int32, "away_score": pl.Int32},
                  null_values=["NA", ""], try_parse_dates=True)
df = df.drop_nulls(subset=["home_score", "away_score"])
max_dt = datetime.strptime(sm.MAX_DATE, "%Y-%m-%d")
df = df.filter((pl.col("date") >= datetime(2018, 1, 1)) & (pl.col("date") <= max_dt))
df = df.with_columns(
    pl.col("home_team").map_elements(sm.normalize_team_name, return_dtype=pl.Utf8),
    pl.col("away_team").map_elements(sm.normalize_team_name, return_dtype=pl.Utf8),
)
df = df.with_columns(
    tournament_weight=pl.col("tournament").replace_strict(sm.TOURNAMENT_WEIGHTS, default=1.0).cast(pl.Float32),
)
df = df.sort("date")

with open(sm.KNOCKOUT_MATCHES) as f:
    matches = json.load(f)

header = f"{'Match':28s} {'H_ELO':>6s} {'A_ELO':>6s} {'EloDiff':>7s} {'H_GF':>6s} {'H_GA':>6s} {'H_Win':>6s} {'H_Form':>5s} {'A_GF':>6s} {'A_GA':>6s} {'A_Win':>6s} {'A_Form':>5s} {'H_H2H':>5s} {'A_H2H':>5s} {'GDiff':>6s}"
print(header)
print("-" * len(header))

for m in matches:
    fv, _, _, _ = sm.build_features_for_match(m, df, sm.MAX_DATE, elo_lookup, static_elo, nlp_data=None)
    print(f'{m["local"]+" vs "+m["visitante"]:28s} '
          f'{fv["home_elo"]:6.0f} {fv["away_elo"]:6.0f} {fv["elo_diff"]:+7.0f} '
          f'{fv["home_goals_for_avg"]:6.3f} {fv["home_goals_against_avg"]:6.3f} '
          f'{fv["home_win_pct"]:6.3f} {fv["home_recent_form"]:5.2f} '
          f'{fv["away_goals_for_avg"]:6.3f} {fv["away_goals_against_avg"]:6.3f} '
          f'{fv["away_win_pct"]:6.3f} {fv["away_recent_form"]:5.2f} '
          f'{fv["h2h_home_wins"]:5d} {fv["h2h_away_wins"]:5d} '
          f'{fv["goal_diff_strength"]:+6.3f}')

print("\n\n=== DIAGNOSTIC: Feature values for Mexico vs Ecuador ===")
for m in matches:
    if m["local"] == "Mexico" and m["visitante"] == "Ecuador":
        fv, _, _, _ = sm.build_features_for_match(m, df, sm.MAX_DATE, elo_lookup, static_elo, nlp_data=None)
        for k, v in fv.items():
            print(f"  {k:35s} = {v}")
        break

print("\n\n=== TRAINING: Raw rolling stats for Mexico ===")
test_date = datetime(2026, 6, 27)
hs = sm.compute_rolling_stats(df, "Mexico", test_date, elo_lookup, static_elo, window_matches=10)
for k, v in hs.items():
    print(f"  {k:25s} = {v}")

print("\n\n=== TRAINING: Raw rolling stats for Ecuador ===")
as_ = sm.compute_rolling_stats(df, "Ecuador", test_date, elo_lookup, static_elo, window_matches=10)
for k, v in as_.items():
    print(f"  {k:25s} = {v}")

print("\n\n=== ELO HISTORY: Mexico's last 5 entries ===")
for row in sm.load_elo_history().get("Mexico", [])[-5:]:
    print(f"  date_ord={row[0]}  elo_before={row[1]:.0f}  elo_after={row[2]:.0f}")

print("\n\n=== ELO HISTORY: Ecuador's last 5 entries ===")
for row in sm.load_elo_history().get("Ecuador", [])[-5:]:
    print(f"  date_ord={row[0]}  elo_before={row[1]:.0f}  elo_after={row[2]:.0f}")

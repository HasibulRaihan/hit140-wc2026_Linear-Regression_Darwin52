"""
MERGE the five clean files into the final Model 2.1 table (104 rows, 8 variables).
Every variable = Team A minus Team B (Team A = home_team column of the match).
Output: features/final_model_2_1_104_rows.csv
"""
import sys
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "clean"
OUT = ROOT / "features"; OUT.mkdir(exist_ok=True)

R1 = pd.read_csv(C / "R1_fifa_ranking.csv")[["team", "fifa_points"]]
R2 = pd.read_csv(C / "R2_squad_value.csv")[["team", "log_market_value"]]
R3 = pd.read_csv(C / "R3_front3_rating.csv")[["team", "front3_rating_mean"]]
R4 = pd.read_csv(C / "R4_form.csv")[["team", "goals_scored_pm", "goals_conceded_pm", "points_pm"]]
R5 = pd.read_csv(C / "R5_match_context.csv")

team = R1.merge(R2, on="team").merge(R3, on="team").merge(R4, on="team")
assert len(team) == 48 and not team.isna().any().any(), "team table incomplete"
VARS = ["fifa_points", "log_market_value", "front3_rating_mean", "goals_scored_pm", "goals_conceded_pm", "points_pm"]

a = team.add_suffix("_a").rename(columns={"team_a": "team_a"})
b = team.add_suffix("_b").rename(columns={"team_b": "team_b"})
m = R5.merge(a, on="team_a", how="left").merge(b, on="team_b", how="left")
assert len(m) == 104 and not m[[v + "_a" for v in VARS]].isna().any().any(), "a team failed to match - check names"

final = m[["match_id", "date", "team_a", "team_b", "goal_diff"]].copy()
for v in VARS:
    final[v + "_diff"] = (m[v + "_a"] - m[v + "_b"]).round(4)
final["momentum_diff"] = m.momentum_diff
final["host_diff"] = m.host_diff
final.to_csv(OUT / "final_model_2_1_104_rows.csv", index=False, encoding="utf-8-sig")
print("rows:", len(final), "| columns:", list(final.columns))
print("missing values:", int(final.isna().sum().sum()))
print("saved features/final_model_2_1_104_rows.csv")

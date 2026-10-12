"""
MERGE the clean files into the final Model 2.2 dataset: 208 rows, 8 explanatory
variables. Target = goals_scored.

  1 own_mean_squad_age             mean age of the team's starters                R1
  2 opp_mean_squad_age             same, opponent                                R1
  3 own_big5_share                 squad share at a big-5 top-flight club        R1
  4 opp_big5_share                 same, opponent                                R1
  5 own_goals_per_match_before     goals per match, earlier matches only          R3
  6 opp_conceded_per_match_before  opponent conceded per match, earlier only      R3
  7 days_rest                      days since this team's previous match          R2
  8 is_knockout                    0 = group, 1 = knockout                        R2

Out: features/final_model_2_2_208_rows.csv

Abishek Rajeshkumar, Rejan Sapkota
"""
import sys
from pathlib import Path

from model_config import X_COLS

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "clean"
OUT = ROOT / "features"
OUT.mkdir(exist_ok=True)



def main():
    profile = pd.read_csv(C / "R1_squad_profile.csv")[["team", "mean_squad_age", "big5_share"]]
    form = pd.read_csv(C / "R3_rolling_form.csv")
    assert len(profile) == 48 and not profile.isna().any().any()

    own = profile.rename(columns={"mean_squad_age": "own_mean_squad_age",
                                  "big5_share": "own_big5_share"})
    opp = profile.rename(columns={"team": "opponent",
                                  "mean_squad_age": "opp_mean_squad_age",
                                  "big5_share": "opp_big5_share"})

    df = form.merge(own, on="team", how="left").merge(opp, on="opponent", how="left")
    assert len(df) == 208, f"row count changed to {len(df)}"
    assert not df[X_COLS].isna().any().any(), "a team failed to match - check names"

    final = df[["match_id", "date", "team", "opponent", "goals_scored"] + X_COLS
               + ["is_first_match"]].copy()
    final[X_COLS] = final[X_COLS].round(4)
    final = final.sort_values(["match_id", "team"]).reset_index(drop=True)
    final.to_csv(OUT / "final_model_2_2_208_rows.csv", index=False, encoding="utf-8-sig")

    print(f"rows {len(final)} | variables {len(X_COLS)} | missing {int(final.isna().sum().sum())} "
          f"| teams {final.team.nunique()} | matches {final.match_id.nunique()}")
    print(final[X_COLS + ["goals_scored"]].describe().T[["mean", "std", "min", "max"]]
          .round(3).to_string())
    print("saved features/final_model_2_2_208_rows.csv")


if __name__ == "__main__":
    main()

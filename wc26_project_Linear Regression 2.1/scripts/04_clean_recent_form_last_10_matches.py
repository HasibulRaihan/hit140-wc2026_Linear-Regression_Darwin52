"""
R4 - Recent form: each team's LAST 10 international matches BEFORE the World Cup (before 11 June 2026).
Input : raw/R4_last10_raw.xlsx  (each team's 10 latest matches before the World Cup, copied unchanged from martj42 results.csv)
Output: clean/R4_last10_matches.csv  
        clean/R4_form.csv            
Only matches played before kickoff are used, so there is no leakage.
"""
import sys
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw" / "R4_last10_raw.xlsx"
R1 = ROOT / "clean" / "R1_fifa_ranking.csv"
OUT = ROOT / "clean"
CUTOFF = "2026-06-11"      # first World Cup match
N = 10


def main():
    r = pd.read_excel(RAW, sheet_name="matches")
    r["date"] = pd.to_datetime(r["date"])
    teams = pd.read_csv(R1).team.tolist()
    assert set(r.team) == set(teams), "team names differ from R1"

    # the raw row is from the home or away side; work out THIS team's goals for / against
    home = r.home_team == r.team
    away = r.away_team == r.team
    assert (home | away).all(), "a row does not involve its team"
    long = r.assign(opponent=r.away_team.where(home, r.home_team),
                    gf=r.home_score.where(home, r.away_score),
                    ga=r.away_score.where(home, r.home_score))[["date", "team", "opponent", "gf", "ga", "tournament"]]
    long["points"] = (long.gf > long.ga) * 3 + (long.gf == long.ga) * 1

    last = (long.sort_values(["team", "date"], ascending=[True, False])
                .groupby("team").head(N))
    n = last.groupby("team").size()
    assert (n == N).all(), f"teams with fewer than {N} matches: {n[n < N].to_dict()}"
    assert last.date.max() < pd.Timestamp(CUTOFF)

    form = last.groupby("team").agg(goals_scored_pm=("gf", "mean"), goals_conceded_pm=("ga", "mean"),
                                    points_pm=("points", "mean"), n_matches=("gf", "size"),
                                    window_start=("date", "min"), window_end=("date", "max")).reset_index()
    form[["goals_scored_pm", "goals_conceded_pm", "points_pm"]] = form[["goals_scored_pm", "goals_conceded_pm", "points_pm"]].round(3)
    assert len(form) == 48 and set(form.team) == set(teams)

    last.sort_values(["team", "date"]).to_csv(OUT / "R4_last10_matches.csv", index=False, encoding="utf-8-sig")
    form.to_csv(OUT / "R4_form.csv", index=False, encoding="utf-8-sig")
    m = form.merge(pd.read_csv(R1), on="team")
    print("teams:", len(form), "| latest match used:", last.date.max().date())
    print("corr points_pm vs FIFA points:", round(m.points_pm.corr(m.fifa_points), 3))
    print(form.sort_values("points_pm", ascending=False).head(5)[["team", "goals_scored_pm", "goals_conceded_pm", "points_pm"]].to_string(index=False))
    print("saved clean/R4_form.csv and clean/R4_last10_matches.csv")


if __name__ == "__main__":
    main()

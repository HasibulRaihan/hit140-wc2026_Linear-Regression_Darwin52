"""
R2 - the 208 team-match rows: 104 matches x 2 teams. Target = goals_scored.
Also builds days_rest and is_knockout.

In : raw/R2_fbref_fixtures_raw.csv  (FBref, 2026 World Cup, Scores & Fixtures)
Out: clean/R2_team_match_rows.csv

Abishek Rajeshkumar, Rejan Sapkota
"""
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw" / "R2_fbref_fixtures_raw.csv"
R1 = ROOT / "clean" / "R1_squad_profile.csv"
OUT = ROOT / "clean"


def strip_country_code(s):
    # The country code trails in the Home column ('Mexico mx') but leads in the Away
    # column ('za South Africa'), so strip both ends.
    s = s.astype(str).str.strip()
    s = s.str.replace(r"^[a-z]{2,3}\s+", "", regex=True)      # 'za South Africa'
    s = s.str.replace(r"\s+[a-z]{2,3}$", "", regex=True)      # 'Mexico mx'
    return s.str.strip()


def main():
    fx = pd.read_csv(RAW)
    fx = fx[fx["Home"].notna() & fx["Away"].notna()]
    fx = fx[fx["Home"].astype(str).str.strip() != "Home"]     # repeated header rows
    fx = fx[fx["Score"].notna() & (fx["Score"].astype(str).str.strip() != "")]

    fx["home"] = strip_country_code(fx["Home"])
    fx["away"] = strip_country_code(fx["Away"])
    fx["date"] = pd.to_datetime(fx["Date"], errors="coerce")

    # Score uses an en dash, and reads '(3) 1-1 (4)' after a shoot-out. The shoot-out
    # score is dropped: the result is recorded at the end of extra time.
    score = (fx["Score"].astype(str)
             .str.replace("–", "-", regex=False)           # en dash -> hyphen
             .str.replace(r"\s*\(\d+\)\s*", "", regex=True))    # drop the shoot-out score
    parts = score.str.extract(r"(\d+)\s*-\s*(\d+)")
    fx["home_goals"] = pd.to_numeric(parts[0], errors="coerce")
    fx["away_goals"] = pd.to_numeric(parts[1], errors="coerce")
    n_shootout = int(fx["Score"].astype(str).str.contains(r"\(\d+\)").sum())

    fx = fx.dropna(subset=["home_goals", "away_goals", "date"]).sort_values("date")
    fx = fx.reset_index(drop=True)
    fx.insert(0, "match_id", range(1, len(fx) + 1))
    assert len(fx) == 104, f"expected 104 matches, found {len(fx)}"

    fx["is_knockout"] = (~fx["Round"].astype(str).str.contains("Group", case=False, na=False)).astype(int)

    # one row per team per match
    home = pd.DataFrame({"match_id": fx.match_id, "date": fx.date, "team": fx.home,
                         "opponent": fx.away, "goals_scored": fx.home_goals,
                         "goals_conceded": fx.away_goals, "is_knockout": fx.is_knockout,
                         "side": "home_listed"})
    away = pd.DataFrame({"match_id": fx.match_id, "date": fx.date, "team": fx.away,
                         "opponent": fx.home, "goals_scored": fx.away_goals,
                         "goals_conceded": fx.home_goals, "is_knockout": fx.is_knockout,
                         "side": "away_listed"})
    rows = pd.concat([home, away]).sort_values(["team", "date"]).reset_index(drop=True)
    rows[["goals_scored", "goals_conceded"]] = rows[["goals_scored", "goals_conceded"]].astype(int)

    # days since that team's previous match
    rows["prev_date"] = rows.groupby("team", sort=False)["date"].shift(1)
    rows["days_rest"] = (rows.date - rows.prev_date).dt.days
    rows["is_first_match"] = rows.days_rest.isna().astype(int)
    median_rest = rows.days_rest.median()
    rows["days_rest"] = rows.days_rest.fillna(median_rest)

    # validation
    assert len(rows) == 208, f"expected 208 rows, found {len(rows)}"
    assert rows.groupby("match_id").size().eq(2).all(), "a match does not have exactly 2 rows"
    assert rows.goals_scored.sum() == fx.home_goals.sum() + fx.away_goals.sum()
    teams = set(pd.read_csv(R1).team)
    assert set(rows.team) == teams, f"team names differ from R1: {sorted(set(rows.team) ^ teams)}"

    rows = rows[["match_id", "date", "team", "opponent", "side", "goals_scored",
                 "goals_conceded", "days_rest", "is_first_match", "is_knockout"]]
    rows.to_csv(OUT / "R2_team_match_rows.csv", index=False, encoding="utf-8-sig")

    print(f"matches        : {len(fx)}  (shoot-outs: {n_shootout})")
    print(f"team-match rows: {len(rows)}  teams: {rows.team.nunique()}  knockout: {int(rows.is_knockout.sum())}")
    print(f"first-match rows (days_rest = median {median_rest:.0f}): {int(rows.is_first_match.sum())}")
    print(f"goals_scored: mean {rows.goals_scored.mean():.3f}, variance {rows.goals_scored.var():.3f}")
    print("distribution:", rows.goals_scored.value_counts().sort_index().to_dict())
    print("saved clean/R2_team_match_rows.csv")


if __name__ == "__main__":
    main()

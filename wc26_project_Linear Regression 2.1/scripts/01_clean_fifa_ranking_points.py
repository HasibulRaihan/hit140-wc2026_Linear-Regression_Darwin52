"""
Clean raw R1 (FIFA ranking, Excel file) -> clean/R1_fifa_ranking.csv (and a .xlsx copy)
Input : raw/R1_fifa_ranking_raw.xlsx   columns: World Cup team | FIFA Rank | FIFA Points
Steps : read the Excel sheet, strip the flag emoji from the team name, fix types,
        standardise team names, validate, save.
"""
import re, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # project root = folder above /scripts
(ROOT / "clean").mkdir(exist_ok=True)

raw = pd.read_excel(ROOT / "raw" / "R1_fifa_ranking_raw.xlsx")
df = pd.DataFrame({
    "team_raw": raw["World Cup team"].astype(str).str.replace(r"^[^\w]+", "", regex=True).str.strip(),  # drop flag emoji
    "fifa_rank": raw["FIFA Rank"].astype(int),
    "fifa_points": raw["FIFA Points"].astype(float),
})

# one team-name standard for the whole project (the spelling used on the match table)
NAME_FIX = {"Türkiye": "Turkey", "Côte d'Ivoire": "Ivory Coast", "Czechia": "Czech Republic"}
df["team"] = df.team_raw.replace(NAME_FIX)

# ---- validation checks ----
assert len(df) == 48, len(df)
assert df.team.is_unique
assert df.fifa_points.between(1000, 2000).all()
assert df.sort_values("fifa_rank").fifa_points.is_monotonic_decreasing   # higher rank <-> more points

out = df[["team", "fifa_rank", "fifa_points", "team_raw"]]
out.to_csv(ROOT / "clean" / "R1_fifa_ranking.csv", index=False)
out.to_excel(ROOT / "clean" / "R1_fifa_ranking.xlsx", index=False)
print(out.head(), "\nrows:", len(out))

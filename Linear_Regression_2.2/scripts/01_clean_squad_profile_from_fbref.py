"""
R1 - squad profile for all 48 teams: mean_squad_age, big5_share.

In : raw/R1_fbref_players_raw.csv   (FBref, 2026 World Cup, Player Standard Stats)
Out: clean/R1_squad_profile.csv

Abishek Rajeshkumar, Rejan Sapkota
"""
import io
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw" / "R1_fbref_players_raw.csv"
OUT = ROOT / "clean"
OUT.mkdir(exist_ok=True)

BIG5 = {"eng", "es", "it", "de", "fr"}
AGE_MIN, AGE_MAX = 16, 45


def read_fbref_csv(path):
    # The export opens with a citation banner and a merged group-header row, so seek
    # the real header instead of skipping a fixed count.
    lines = Path(path).read_text(encoding="utf-8-sig", errors="replace").split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("Rk,"))
    return pd.read_csv(io.StringIO("\n".join(lines[start:])))


def strip_country_code(s):
    # 'us USA' -> 'USA'
    return (s.astype(str).str.strip()
            .str.replace(r"^[a-z]{2,3}\s+", "", regex=True).str.strip())


def main():
    df = read_fbref_csv(RAW)
    df = df[df["Player"] != "Player"]                 # header repeats every 25 rows
    before = len(df)
    df = df.drop_duplicates(["Player", "Squad"])
    df["team"] = strip_country_code(df["Squad"])

    # Club reads '1.eng Leeds United': leading digit = division tier, letters = country.
    club = df["Club"].astype(str).str.extract(r"^(\d+)\.([a-z]+)")
    df["tier"] = pd.to_numeric(club[0], errors="coerce")
    df["club_country"] = club[1]
    df["at_big5"] = ((df.tier == 1) & (df.club_country.isin(BIG5))).astype(int)

    df["age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["starts"] = pd.to_numeric(df["Starts"], errors="coerce").fillna(0)

    starters = df[(df.starts >= 1) & df.age.between(AGE_MIN, AGE_MAX)]

    print(f"players read              : {before}")
    print(f"duplicates dropped        : {before - len(df)}")
    print(f"missing age               : {int(df.age.isna().sum())}")
    print(f"age outside {AGE_MIN}-{AGE_MAX}         : {int((~df.age.between(AGE_MIN, AGE_MAX)).sum())}")
    print(f"club not parsed           : {int(df.tier.isna().sum())}")
    print(f"at a big-5 top-flight club: {int(df.at_big5.sum())}")
    print(f"starters (>= 1 start)     : {len(starters)}")

    profile = (df.groupby("team")
                 .agg(squad_size=("Player", "count"), n_big5=("at_big5", "sum"),
                      big5_share=("at_big5", "mean"))
                 .join(starters.groupby("team").agg(mean_squad_age=("age", "mean"),
                                                    n_starters=("Player", "count")))
                 .round(4).reset_index())

    assert len(profile) == 48, f"expected 48 squads, built {len(profile)}"
    assert profile.team.is_unique
    assert profile.big5_share.between(0, 1).all()
    assert profile.mean_squad_age.between(20, 35).all(), "implausible squad age"
    assert profile.notna().all().all(), "a squad has no starters"

    profile.to_csv(OUT / "R1_squad_profile.csv", index=False, encoding="utf-8-sig")

    r = profile.mean_squad_age.corr(profile.big5_share)
    print(f"corr(mean_squad_age, big5_share) = {r:+.3f}")
    print("saved clean/R1_squad_profile.csv")


if __name__ == "__main__":
    main()

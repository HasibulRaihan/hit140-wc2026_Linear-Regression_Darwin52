"""
R3 - CLEAN the OVRBase raw file into the 'front three rating' variable.
Front three = the 3 highest-rated attacking players of the squad (ST, CF, LW, RW, LM, RM, CAM).
If a squad has fewer than 3 rated attackers: fill with best CM/CDM, then with DEFAULT_RATING (60).
Input : raw/R3_ovrbase_raw.xlsx     Output: clean/R3_front3_players_rated.csv, clean/R3_front3_rating.csv
"""
import sys
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw" / "R3_ovrbase_raw.xlsx"
R1 = ROOT / "clean" / "R1_fifa_ranking.csv"
OUT = ROOT / "clean"
ATTACK = ["ST", "CF", "LW", "RW", "LM", "RM", "CAM"]
MID = ["CM", "CDM"]
DEFAULT_RATING = 60   # used only for a missing slot (player not in EA FC 26); stated in the report


def select_front3(g):
    g = g.sort_values("ovr", ascending=False)
    a = g[g.pos.isin(ATTACK)].head(3).assign(source="attacker")
    if len(a) < 3:
        m = g[g.pos.isin(MID)].head(3 - len(a)).assign(source="midfielder")
        a = pd.concat([a, m])
    return a


def main():
    if not RAW.exists():
        sys.exit(f"Missing {RAW}")
    df = pd.read_excel(RAW, sheet_name="rated")
    un = pd.read_excel(RAW, sheet_name="unrated_forwards")
    df["team"] = df.team.str.strip(); df["pos"] = df.pos.str.strip().str.upper()
    teams = set(pd.read_csv(R1).team)
    assert set(df.team) == teams, f"team names differ: {sorted(set(df.team) ^ teams)}"
    assert df.ovr.between(30, 99).all(), "rating outside 30-99"
    assert not df.duplicated(["team", "player"]).any(), "duplicate player"

    parts = []
    for t, g in df.groupby("team"):
        s = select_front3(g).copy(); s["team"] = t
        parts.append(s)
    players = pd.concat(parts)
    rows = []
    for t, s in players.groupby("team"):
        vals = list(s.ovr)
        n_default = 3 - len(vals)
        vals += [DEFAULT_RATING] * n_default
        rows.append(dict(team=t, front3_rating_mean=round(sum(vals) / 3, 2), front3_rating_max=max(vals),
                         front3_players="; ".join(f"{p} ({o})" for p, o in zip(s.player, s.ovr)),
                         n_midfielder_used=int((s.source == "midfielder").sum()), n_default_used=n_default))
    f3 = pd.DataFrame(rows).merge(un.rename(columns={"unrated_forwards": "n_unrated_forwards"}), on="team")
    assert len(f3) == 48
    players[["team", "player", "pos", "ovr", "source"]].to_csv(OUT / "R3_front3_players_rated.csv", index=False, encoding="utf-8-sig")
    f3.to_csv(OUT / "R3_front3_rating.csv", index=False, encoding="utf-8-sig")
    m = f3.merge(pd.read_csv(R1), on="team")
    print("teams:", len(f3), "| correlation with FIFA points:", round(m.front3_rating_mean.corr(m.fifa_points), 3))
    print("squads with a default/midfielder slot:", f3[(f3.n_default_used > 0) | (f3.n_midfielder_used > 0)].team.tolist())
    print("saved clean/R3_front3_rating.csv")


if __name__ == "__main__":
    main()

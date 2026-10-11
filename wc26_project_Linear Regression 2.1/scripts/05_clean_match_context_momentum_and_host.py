"""
R5 - Match context variables (all known BEFORE kickoff) for the 104 World Cup 2026 matches.
  momentum_diff = Team A - Team B 
     momentum = average goal difference in the team's EARLIER World Cup matches (earlier dates only).
     First match of a team (no earlier WC game): use its pre-tournament goal difference per match from R4
     (goals scored - goals conceded, last 10 matches). No imputation, no guessing.
  host_diff     = host_A - host_B   (host = Mexico / Canada / United States playing in its own country)
Input : raw/R5_wc2026_matches_raw.xlsx, clean/R4_form.csv
Output: clean/R5_match_context.csv  (one row per match, target = goal_diff = Team A goals - Team B goals)
"""
import sys
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw" / "R5_wc2026_matches_raw.xlsx"
OUT = ROOT / "clean"
HOSTS = {"Mexico", "Canada", "United States"}


def main():
    m = pd.read_excel(RAW, sheet_name="matches")
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["date", "home_team"]).reset_index(drop=True)
    assert len(m) == 104, f"expected 104 matches, found {len(m)}"
    assert not m[["home_score", "away_score"]].isna().any().any()
    m.insert(0, "match_id", range(1, 105))
    r4 = pd.read_csv(OUT / "R4_form.csv").set_index("team")
    pre = r4.goals_scored_pm - r4.goals_conceded_pm          # pre-tournament goal difference per match

    def momentum(team, date):
        gd = []
        for _, x in m[m.date < date].iterrows():           # strictly earlier days only
            if x.home_team == team:
                gd.append(x.home_score - x.away_score)
            elif x.away_team == team:
                gd.append(x.away_score - x.home_score)
        return (sum(gd) / len(gd), len(gd)) if gd else (float(pre[team]), 0)

    rows = []
    for _, x in m.iterrows():
        ma, na = momentum(x.home_team, x.date)
        mb, nb = momentum(x.away_team, x.date)
        ha = int(x.home_team in HOSTS and x.country == x.home_team)
        hb = int(x.away_team in HOSTS and x.country == x.away_team)
        rows.append(dict(match_id=x.match_id, date=x.date.date(), team_a=x.home_team, team_b=x.away_team,
                         goal_diff=int(x.home_score - x.away_score),
                         momentum_a=round(ma, 3), momentum_b=round(mb, 3), momentum_diff=round(ma - mb, 3),
                         n_prior_a=na, n_prior_b=nb, host_a=ha, host_b=hb, host_diff=ha - hb))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "R5_match_context.csv", index=False, encoding="utf-8-sig")
    print("matches:", len(out), "| host matches:", int((out.host_diff != 0).sum()),
          "| rows using pre-tournament fallback (team's 1st game):", int((out.n_prior_a == 0).sum() + (out.n_prior_b == 0).sum()))
    print("corr momentum_diff vs goal_diff:", round(out.momentum_diff.corr(out.goal_diff), 3),
          "| corr host_diff vs goal_diff:", round(out.host_diff.corr(out.goal_diff), 3))
    print("saved clean/R5_match_context.csv")


if __name__ == "__main__":
    main()

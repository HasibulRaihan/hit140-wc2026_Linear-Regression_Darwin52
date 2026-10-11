"""
R3 - in-tournament form computed from earlier matches only (leakage-safe).

  own_goals_per_match_before     team's goals per match in its earlier matches
  opp_conceded_per_match_before  opponent's goals conceded per match in theirs

Three steps: shift-based rolling means, an independent re-derivation of 40 random
rows that raises on mismatch, and a deliberate fit of the leaky version to measure
the inflation. First matches take 0 and are flagged with is_first_match.

In : clean/R2_team_match_rows.csv
Out: clean/R3_rolling_form.csv, clean/R3_leakage_check.csv

Abishek Rajeshkumar, Rejan Sapkota
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "clean"
N_VERIFY = 40


def add_rolling_form(rows):
    # Per team in date order: cumulative total minus the current row, over matches
    # already played. Subtracting the current row is what makes it safe.
    rows = rows.sort_values(["team", "date"]).copy()
    g = rows.groupby("team", sort=False)

    rows["matches_before"] = g.cumcount()
    scored_before = g["goals_scored"].cumsum() - rows["goals_scored"]
    conceded_before = g["goals_conceded"].cumsum() - rows["goals_conceded"]

    n = rows["matches_before"].replace(0, np.nan)
    rows["own_goals_per_match_before"] = (scored_before / n).fillna(0.0)
    rows["own_conceded_per_match_before"] = (conceded_before / n).fillna(0.0)
    return rows.sort_values(["match_id", "team"]).reset_index(drop=True)


def attach_opponent_form(rows):
    # Read from the opponent's own row in the same match, so the same rule applies.
    key = rows.set_index(["match_id", "team"])["own_conceded_per_match_before"]
    rows["opp_conceded_per_match_before"] = [
        float(key.get((m, o), 0.0)) for m, o in zip(rows.match_id, rows.opponent)]
    return rows


def verify_no_leakage(rows, n=N_VERIFY):
    """Re-derive a sample of rows by date filtering; raise on any mismatch."""
    bad = []
    for i in rows.sample(n=min(n, len(rows)), random_state=42).index:
        r = rows.loc[i]
        earlier = rows[(rows.team == r.team) & (rows.date < r.date)]
        expected = earlier.goals_scored.mean() if len(earlier) else 0.0
        if abs(expected - r.own_goals_per_match_before) > 1e-9:
            bad.append((r.team, str(r.date)[:10], expected, r.own_goals_per_match_before))
    if bad:
        raise ValueError(f"LEAKAGE DETECTED in {len(bad)} sampled rows - do not proceed: {bad[:3]}")
    print(f"  {n} rows re-derived independently: all match. No leakage.")


def quantify_the_trap(rows):
    """Fit the leaky version on purpose and measure the inflation."""
    X_safe = rows[["own_goals_per_match_before", "opp_conceded_per_match_before"]]
    leaky = rows.copy()
    leaky["own_goals_per_match_before"] = leaky.groupby("team").goals_scored.transform("mean")
    leaky["opp_conceded_per_match_before"] = leaky.opponent.map(
        rows.groupby("team").goals_conceded.mean())
    X_leak = leaky[["own_goals_per_match_before", "opp_conceded_per_match_before"]]
    y = rows.goals_scored
    r_safe = sm.OLS(y, sm.add_constant(X_safe)).fit().rsquared
    r_leak = sm.OLS(y, sm.add_constant(X_leak)).fit().rsquared
    print("  these two form variables alone:")
    print(f"    correct (earlier matches only) R2 = {r_safe:.4f}")
    print(f"    leaky   (tournament totals)   R2 = {r_leak:.4f}")
    print(f"    inflation {r_leak - r_safe:+.4f}  ({r_leak / max(r_safe, 1e-9):.1f}x)")
    return dict(r2_correct=round(r_safe, 4), r2_leaky=round(r_leak, 4),
                inflation=round(r_leak - r_safe, 4))


def main():
    rows = pd.read_csv(C / "R2_team_match_rows.csv", parse_dates=["date"])
    rows = add_rolling_form(rows)
    rows = attach_opponent_form(rows)

    print("LEAKAGE VERIFICATION")
    verify_no_leakage(rows)
    trap = quantify_the_trap(rows)
    pd.DataFrame([trap]).to_csv(C / "R3_leakage_check.csv", index=False)

    out = rows[["match_id", "date", "team", "opponent", "goals_scored", "matches_before",
                "own_goals_per_match_before", "own_conceded_per_match_before",
                "opp_conceded_per_match_before", "days_rest", "is_first_match", "is_knockout"]]
    assert len(out) == 208
    assert not out.isna().any().any()
    assert (out.loc[out.is_first_match == 1, "own_goals_per_match_before"] == 0).all()
    out.to_csv(C / "R3_rolling_form.csv", index=False, encoding="utf-8-sig")

    print(f"  rows with no prior match (0, flagged): {int(out.is_first_match.sum())}")
    print(f"  own_goals_per_match_before   : mean {out.own_goals_per_match_before.mean():.3f}")
    print(f"  opp_conceded_per_match_before: mean {out.opp_conceded_per_match_before.mean():.3f}")
    print("saved clean/R3_rolling_form.csv and clean/R3_leakage_check.csv")


if __name__ == "__main__":
    main()

"""
Predicted goals vs the actual World Cup 2026 results. Four checks:
  A) fitted on all 208 rows
  B) leave-one-MATCH-out (both rows of a match held out together)
  C) train on the 144 group-stage rows -> predict the 64 knockout rows
  D) baseline: predict the training mean
Reported in football terms too: within one goal, and the over/under 1.5 call.

In : features/final_model_2_2_208_rows.csv
Out: features/model_2_2_predictions_vs_actual.csv / .xlsx,
     model_2_2_evaluation_summary.csv, model_2_2_team_accuracy.csv,
     figures/predicted_vs_actual_by_team_match.png, figures/team_level_story.png

Abishek Rajeshkumar, Rejan Sapkota
"""
import sys
from pathlib import Path

from model_config import X_COLS
from model_data import load_model_data
from model_metrics import regression_metrics

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def metrics(name, actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    scores = regression_metrics(actual, pred)
    rmse, mae, r2 = scores["RMSE"], scores["MAE"], scores["R2"]
    within1 = np.mean(np.abs(actual - pred) <= 1)
    ou = np.mean((pred > 1.5) == (actual > 1.5))          # over / under 1.5 goals
    print(f"{name:36s} n={len(actual):3d}  RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:+.3f}  "
          f"within 1 goal={within1:.0%}  over/under 1.5={ou:.0%}")
    return dict(check=name, n=len(actual), RMSE=round(rmse, 3), MAE=round(mae, 3),
                R2=round(r2, 3), within_1_goal=round(within1, 3), over_under_1_5=round(ou, 3))


def main():
    ROOT = Path(__file__).resolve().parent.parent
    F = ROOT / "features"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
    df = load_model_data()
    X, y = df[X_COLS], df.goals_scored




    # A) fitted on everything
    fit_all = LinearRegression().fit(X, y).predict(X)

    # B) leave-one-MATCH-out: the two rows of a match are paired
    loo = np.empty(len(df))
    for m in df.match_id.unique():
        te = (df.match_id == m).values
        loo[te] = LinearRegression().fit(X[~te], y[~te]).predict(X[te])

    # C) group stage -> knockout
    ko = (df.is_knockout == 1).values
    ko_pred = LinearRegression().fit(X[~ko], y[~ko]).predict(X[ko])

    res = [metrics("A) fitted on all 208", y, fit_all),
           metrics("B) leave-one-match-out (all 208)", y, loo),
           metrics("C) group(144) -> knockout(64)", y[ko], ko_pred),
           metrics("D) baseline: group-stage mean", y[ko], np.full(ko.sum(), y[~ko].mean()))]
    pd.DataFrame(res).to_csv(F / "model_2_2_evaluation_summary.csv", index=False)

    out = df[["match_id", "date", "team", "opponent", "goals_scored"]].rename(
        columns={"goals_scored": "actual_goals"}).copy()
    out["predicted_loo"] = loo.round(2)
    out["error_loo"] = (out.actual_goals - out.predicted_loo).round(2)
    out["within_1_goal"] = out.error_loo.abs() <= 1
    out.to_csv(F / "model_2_2_predictions_vs_actual.csv", index=False, encoding="utf-8-sig")
    out.to_excel(F / "model_2_2_predictions_vs_actual.xlsx", index=False)

    print(f"\nPredictions within one goal of the truth: {out.within_1_goal.mean():.0%}")
    worst = out.reindex(out.error_loo.abs().sort_values(ascending=False).index).head(3)
    print("biggest misses:", ", ".join(
        f"{r.team} v {r.opponent} (scored {int(r.actual_goals)}, predicted {r.predicted_loo:.1f})"
        for r in worst.itertuples()))

    # ---- team level ----
    team = (out.groupby("team")
               .agg(matches=("actual_goals", "size"), actual=("actual_goals", "sum"),
                    predicted=("predicted_loo", "sum"))
               .assign(difference=lambda d: (d.actual - d.predicted).round(2))
               .sort_values("difference", ascending=False).reset_index())
    team.to_csv(F / "model_2_2_team_accuracy.csv", index=False, encoding="utf-8-sig")
    print("\nMost UNDER-rated (scored more than predicted):",
          ", ".join(f"{r.team} {r.difference:+.1f}" for r in team.head(3).itertuples()))
    print("Most OVER-rated  (scored fewer than predicted):",
          ", ".join(f"{r.team} {r.difference:+.1f}" for r in team.tail(3).itertuples()))

    # ---- figure 1 ----
    plt.figure(figsize=(6.4, 6))
    jitter = np.random.default_rng(0).normal(0, .07, len(y))
    plt.scatter(loo, y + jitter, s=26, alpha=.5, color="#4a6fa5", edgecolor="white", linewidth=.4)
    plt.plot([-.5, 7.5], [-.5, 7.5], "k--", lw=1.3, label="perfect prediction")
    plt.axvline(0, color="#c0504d", lw=1.2)
    plt.xlabel("predicted goals (leave-one-match-out)")
    plt.ylabel("actual goals scored (jittered to show overlap)")
    plt.title("Model 2.2: predicted vs actual, 208 team-matches", fontweight="bold")
    plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(FIG / "predicted_vs_actual_by_team_match.png", dpi=200); plt.close()

    # ---- figure 2 ----
    t = team.sort_values("difference")
    fig, ax = plt.subplots(figsize=(9, 11))
    cols = ["#c0504d" if d < 0 else "#2e7d64" for d in t.difference]
    ax.barh(t.team, t.difference, color=cols, edgecolor="black", linewidth=.4)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("actual tournament goals minus predicted  (negative = scored fewer than the model expected)")
    ax.set_title("Which teams the pre-match model got wrong", fontweight="bold")
    ax.grid(axis="x", alpha=.25)
    lo, hi = t.iloc[0], t.iloc[-1]
    span = max(abs(t.difference.min()), t.difference.max())
    ax.set_xlim(-span * 1.75, span * 1.75)
    # annotate in the empty half so labels never sit over a bar
    ax.annotate(f"{hi.team} outscored the model\nby {hi.difference:+.1f} goals over {int(hi.matches)} matches",
                xy=(hi.difference, len(t) - 1), xytext=(-span * 1.65, len(t) - 3.2),
                fontsize=9.5, color="#2e7d64", va="center",
                arrowprops=dict(arrowstyle="->", color="#2e7d64", lw=1.2))
    ax.annotate(f"{lo.team} scored {abs(lo.difference):.1f} goals FEWER\nthan the model predicted",
                xy=(lo.difference, 0), xytext=(span * 0.30, 2.6),
                fontsize=9.5, color="#c0504d", va="center",
                arrowprops=dict(arrowstyle="->", color="#c0504d", lw=1.2))
    ax.text(span * 1.68, len(t) - 1, "scored MORE\nthan predicted", ha="right", va="center",
            fontsize=9, color="#2e7d64", style="italic")
    ax.text(-span * 1.68, 0, "scored FEWER\nthan predicted", ha="left", va="center",
            fontsize=9, color="#c0504d", style="italic")
    fig.tight_layout(); fig.savefig(FIG / "team_level_story.png", dpi=200); plt.close(fig)

    print("\nsaved predictions, evaluation summary, team accuracy and 2 figures")


if __name__ == "__main__":
    main()

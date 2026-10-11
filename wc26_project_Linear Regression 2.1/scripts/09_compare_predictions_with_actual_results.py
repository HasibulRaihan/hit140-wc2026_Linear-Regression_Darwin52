"""
Compare the model's predicted goal difference with the ACTUAL World Cup 2026 results.
Three checks:
  A) Fitted on all 104 matches (how well the model explains the tournament)
  B) Leave-one-out: each match is predicted by a model that never saw that match (honest, all 104)
  C) Train on the 72 group matches -> predict the 32 knockout matches (a real "future" test)
Input : features/final_model_2_1_104_rows.csv
Output: features/model_2_1_predictions_vs_actual.csv / .xlsx, figures/predicted_vs_actual_by_match.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "features"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
df = pd.read_csv(F / "final_model_2_1_104_rows.csv")
X_COLS = ["fifa_points_diff", "log_market_value_diff", "front3_rating_mean_diff", "goals_scored_pm_diff",
          "goals_conceded_pm_diff", "points_pm_diff", "momentum_diff", "host_diff"]
X, y = df[X_COLS], df.goal_diff


def metrics(name, actual, pred):
    rmse = np.sqrt(np.mean((actual - pred) ** 2)); mae = np.mean(np.abs(actual - pred))
    r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)
    nd = actual != 0   # winner check ignores draws (no winner to predict)
    win = np.mean(np.sign(pred[nd]) == np.sign(actual[nd]))
    print(f"{name:34s} n={len(actual):3d}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.2f}  winner correct (non-draws)={win:.0%}")
    return dict(check=name, n=len(actual), RMSE=round(rmse, 3), MAE=round(mae, 3), R2=round(r2, 3), winner_accuracy=round(win, 3))


fit_all = LinearRegression().fit(X, y).predict(X)
loo = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
ko = (df.match_id > 72).values      # matches 73-104 are the knockout rounds (sorted by date)
ko_pred = LinearRegression().fit(X[~ko], y[~ko]).predict(X[ko])

res = [metrics("A) fitted on all 104", y.values, fit_all),
       metrics("B) leave-one-out (all 104)", y.values, loo),
       metrics("C) group(72) -> knockout(32)", y[ko].values, ko_pred),
       metrics("baseline: always predict group mean", y[ko].values, np.full(ko.sum(), y[~ko].mean()))]
pd.DataFrame(res).to_csv(F / "model_2_1_evaluation_summary.csv", index=False)

out = df[["match_id", "date", "team_a", "team_b", "goal_diff"]].rename(columns={"goal_diff": "actual_goal_diff"}).copy()
out["predicted_loo"] = loo.round(2)
out["error_loo"] = (out.actual_goal_diff - out.predicted_loo).round(2)
out["predicted_winner"] = np.where(loo > 0.25, out.team_a, np.where(loo < -0.25, out.team_b, "Draw"))
out["actual_winner"] = np.where(y > 0, out.team_a, np.where(y < 0, out.team_b, "Draw"))
out["winner_correct"] = out.predicted_winner == out.actual_winner
out.to_csv(F / "model_2_1_predictions_vs_actual.csv", index=False, encoding="utf-8-sig")
out.to_excel(F / "model_2_1_predictions_vs_actual.xlsx", index=False)
print("\nthree-way outcome (A win / draw / B win) correct:", f"{out.winner_correct.mean():.0%}",
      "| biggest misses:", ", ".join(f"{r.team_a}-{r.team_b} ({r.actual_goal_diff:+d} vs {r.predicted_loo:+.1f})"
                                    for r in out.reindex(out.error_loo.abs().sort_values(ascending=False).index).head(3).itertuples()))

plt.figure(figsize=(6, 6)); plt.scatter(loo, y, s=16, alpha=.6); plt.plot([-6, 6], [-6, 6], "k--")
plt.xlabel("predicted goal difference (leave-one-out)"); plt.ylabel("actual goal difference (Team A - Team B)")
plt.title("Model 2.1: predicted vs actual, 104 matches"); plt.tight_layout()
plt.savefig(FIG / "predicted_vs_actual_by_match.png", dpi=150)
print("saved features/model_2_1_predictions_vs_actual.csv/.xlsx and evaluation summary")

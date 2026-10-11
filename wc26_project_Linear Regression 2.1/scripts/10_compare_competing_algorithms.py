"""
Compare competing regression algorithms for Model 2.1 (HD requirement), all on the same 104 matches and 8 variables.
Algorithms (the two non-linear ones named by the teacher): Linear regression, Random Forest, Support Vector Regression (SVR).
Evaluation (A): 5-fold CV repeated 20 times -> mean RMSE / MAE / R2 (+ spread)
Evaluation (B): train on the 72 group matches -> predict the 32 knockout matches (a real "future" test)
Baseline: always predict the training mean.   Input: features/final_model_2_1_104_rows.csv
Output: features/algorithm_comparison_cv.csv, features/algorithm_comparison_knockout.csv, figures/algorithm_comparison.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
S = StandardScaler
models = {
    "Baseline (mean)": DummyRegressor(strategy="mean"),
    "Linear regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0),
    "SVR (RBF)": make_pipeline(S(), SVR(C=3, epsilon=0.5)),
}
cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
rows = []
for name, m in models.items():
    r = cross_validate(clone(m), X, y, cv=cv, scoring=scoring)
    rows.append(dict(model=name, RMSE=-r["test_rmse"].mean(), RMSE_sd=r["test_rmse"].std(), MAE=-r["test_mae"].mean(), R2=r["test_r2"].mean()))
cvres = pd.DataFrame(rows).round(3).sort_values("RMSE")
cvres.to_csv(F / "algorithm_comparison_cv.csv", index=False)
print("A) repeated 5-fold CV (20 repeats), 104 matches\n", cvres.to_string(index=False))

tr, te = (df.match_id <= 72).values, (df.match_id > 72).values
rows = []
for name, m in models.items():
    p = clone(m).fit(X[tr], y[tr]).predict(X[te])
    rows.append(dict(model=name, RMSE=np.sqrt(mean_squared_error(y[te], p)), MAE=mean_absolute_error(y[te], p), R2=r2_score(y[te], p)))
ko = pd.DataFrame(rows).round(3).sort_values("RMSE")
ko.to_csv(F / "algorithm_comparison_knockout.csv", index=False)
print("\nB) trained on 72 group matches -> predicting 32 knockout matches\n", ko.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
order = cvres.sort_values("RMSE", ascending=False).model.tolist()
for a, t, d, err in [(ax[0], "Repeated 5-fold CV (104 matches)", cvres, True), (ax[1], "Group stage -> knockout (32 matches)", ko, False)]:
    d = d.set_index("model").loc[order]
    colors = ["#999999" if m.startswith("Baseline") else ("#1b6ca8" if m == "Linear regression" else "#9ec5e6") for m in d.index]
    a.barh(d.index, d.RMSE, xerr=d.RMSE_sd if err else None, color=colors, capsize=2)
    a.set_title(t, fontsize=10); a.set_xlabel("RMSE (goals) - lower is better")
    for i, v in enumerate(d.RMSE): a.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "algorithm_comparison.png", dpi=170)
print("saved features/algorithm_comparison_*.csv and figures/algorithm_comparison.png")

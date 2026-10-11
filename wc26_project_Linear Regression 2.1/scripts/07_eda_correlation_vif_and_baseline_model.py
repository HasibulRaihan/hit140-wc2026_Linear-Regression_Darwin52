"""
EDA (why each variable) + Linear Regression 2.1 (OLS) + cross-validation.
Input : features/final_model_2_1_104_rows.csv
Output: features/EDA_correlations.csv, features/EDA_vif.csv, features/model_2_1_coefficients.csv,
        features/model_2_1_summary.txt, figures/*.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_predict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "features"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
df = pd.read_csv(F / "final_model_2_1_104_rows.csv")
X_COLS = ["fifa_points_diff", "log_market_value_diff", "front3_rating_mean_diff", "goals_scored_pm_diff",
          "goals_conceded_pm_diff", "points_pm_diff", "momentum_diff", "host_diff"]
assert len(X_COLS) == 8
y = df.goal_diff

# 1) EDA: correlation with the target
corr = df[X_COLS + ["goal_diff"]].corr()
corr["goal_diff"].drop("goal_diff").rename("pearson_r_with_goal_diff").to_csv(F / "EDA_correlations.csv")
print("Correlation with goal_diff:\n", corr["goal_diff"].drop("goal_diff").round(3).to_string())

# 2) EDA: multicollinearity (VIF)
Xc = sm.add_constant(df[X_COLS])
vif = pd.DataFrame({"variable": X_COLS, "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(len(X_COLS))]})
vif.to_csv(F / "EDA_vif.csv", index=False)
print("\nVIF (above 5-10 = overlap between variables):\n", vif.round(2).to_string(index=False))

# 3) scatterplots + heatmap
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for a, c in zip(ax.ravel(), X_COLS):
    a.scatter(df[c], y, s=14, alpha=.6); a.set_xlabel(c); a.set_ylabel("goal_diff")
    a.set_title(f"r = {corr.loc[c, 'goal_diff']:.2f}")
plt.tight_layout(); plt.savefig(FIG / "scatter_vs_goal_diff.png", dpi=150); plt.close()
plt.figure(figsize=(8, 6.5)); plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7); plt.yticks(range(len(corr)), corr.columns, fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "correlation_heatmap.png", dpi=150); plt.close()

# 4) the model (OLS, standardised so coefficients are comparable)
Z = (df[X_COLS] - df[X_COLS].mean()) / df[X_COLS].std()
ols = sm.OLS(y, sm.add_constant(Z)).fit()
(F / "model_2_1_summary.txt").write_text(str(ols.summary()), encoding="utf-8")
pd.DataFrame({"coef_per_1sd": ols.params, "p_value": ols.pvalues}).round(4).to_csv(F / "model_2_1_coefficients.csv")
print("\nOLS (standardised)  R2 =", round(ols.rsquared, 3), "| adj R2 =", round(ols.rsquared_adj, 3), "| F p =", round(ols.f_pvalue, 4))
print(pd.DataFrame({"coef": ols.params, "p": ols.pvalues}).round(3).to_string())

# 5) honest out-of-sample check (repeated 5-fold CV) vs a "predict the average" baseline
rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)
rm, ma, bl = [], [], []
for tr, te in rkf.split(df):
    mdl = LinearRegression().fit(df.loc[tr, X_COLS], y[tr]); p = mdl.predict(df.loc[te, X_COLS])
    rm.append(np.sqrt(np.mean((y[te] - p) ** 2))); ma.append(np.mean(np.abs(y[te] - p)))
    bl.append(np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2)))
print(f"\nCV RMSE = {np.mean(rm):.3f} | CV MAE = {np.mean(ma):.3f} | baseline RMSE (always predict mean) = {np.mean(bl):.3f}")

# 6) actual vs predicted plot
pred = cross_val_predict(LinearRegression(), df[X_COLS], y, cv=5)
plt.figure(figsize=(5, 5)); plt.scatter(pred, y, s=14, alpha=.6); lim = [-6, 6]; plt.plot(lim, lim, "k--")
plt.xlabel("predicted goal difference (cross-validated)"); plt.ylabel("actual goal difference")
plt.tight_layout(); plt.savefig(FIG / "actual_vs_predicted.png", dpi=150); plt.close()
print("saved EDA csvs, model files and figures")

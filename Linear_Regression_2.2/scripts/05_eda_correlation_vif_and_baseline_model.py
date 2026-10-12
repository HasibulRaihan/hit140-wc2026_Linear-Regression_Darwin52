"""
EDA (correlations, VIF) + the baseline OLS model + cross-validation.

Input : features/final_model_2_2_208_rows.csv
Output: features/EDA_correlations.csv, features/EDA_vif.csv,
        features/model_2_2_coefficients.csv, features/model_2_2_summary.txt,
        figures/scatter_vs_goals_scored.png, figures/correlation_heatmap.png,
        figures/actual_vs_predicted.png

Authors: Abishek Rajeshkumar, Rejan Sapkota
"""
import sys
from pathlib import Path

from model_config import X_COLS

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_predict, GroupKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "features"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
df = pd.read_csv(F / "final_model_2_2_208_rows.csv")
assert len(X_COLS) == 8
NICE = {"own_mean_squad_age": "own mean squad age", "opp_mean_squad_age": "opponent mean squad age",
        "own_big5_share": "own big-5 squad share", "opp_big5_share": "opponent big-5 squad share",
        "own_goals_per_match_before": "own goals p/m, earlier matches",
        "opp_conceded_per_match_before": "opponent conceded p/m, earlier matches",
        "days_rest": "days of rest", "is_knockout": "knockout stage"}
y = df.goals_scored

# 1) EDA: correlation with the target
corr = df[X_COLS + ["goals_scored"]].corr()
corr["goals_scored"].drop("goals_scored").rename("pearson_r_with_goals_scored").to_csv(F / "EDA_correlations.csv")
print("Correlation with goals_scored:\n", corr["goals_scored"].drop("goals_scored").round(3).to_string())

# 2) EDA: multicollinearity (VIF)
Xc = sm.add_constant(df[X_COLS])
vif = pd.DataFrame({"variable": X_COLS,
                    "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(len(X_COLS))]})
vif.to_csv(F / "EDA_vif.csv", index=False)
print("\nVIF:\n", vif.round(2).to_string(index=False))

# 3) scatterplots + heatmap
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for a, c in zip(ax.ravel(), X_COLS):
    a.scatter(df[c], y, s=14, alpha=.5)
    a.set_xlabel(NICE[c]); a.set_ylabel("goals scored")
    a.set_title(f"r = {corr.loc[c, 'goals_scored']:.2f}")
plt.tight_layout(); plt.savefig(FIG / "scatter_vs_goals_scored.png", dpi=150); plt.close()

plt.figure(figsize=(8.5, 7))
plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); plt.colorbar()
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6,
                 color="white" if abs(corr.iloc[i, j]) > .55 else "black")
plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr)), corr.columns, fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "correlation_heatmap.png", dpi=150); plt.close()

# 4) OLS, standardised so coefficients are comparable
Z = (df[X_COLS] - df[X_COLS].mean()) / df[X_COLS].std()
ols = sm.OLS(y, sm.add_constant(Z)).fit()
(F / "model_2_2_summary.txt").write_text(str(ols.summary()), encoding="utf-8")
pd.DataFrame({"coef_per_1sd": ols.params, "p_value": ols.pvalues}).round(4).to_csv(F / "model_2_2_coefficients.csv")
print("\nOLS (standardised)  R2 =", round(ols.rsquared, 3), "| adj R2 =", round(ols.rsquared_adj, 3),
      "| F p =", f"{ols.f_pvalue:.2e}")
print(pd.DataFrame({"coef": ols.params, "p": ols.pvalues}).round(3).to_string())

# 5) repeated 5-fold CV vs a predict-the-mean baseline
rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)
rm, ma, bl = [], [], []
for tr, te in rkf.split(df):
    mdl = LinearRegression().fit(df.loc[tr, X_COLS], y[tr]); p = mdl.predict(df.loc[te, X_COLS])
    rm.append(np.sqrt(np.mean((y[te] - p) ** 2))); ma.append(np.mean(np.abs(y[te] - p)))
    bl.append(np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2)))
print(f"\nCV RMSE = {np.mean(rm):.3f} | CV MAE = {np.mean(ma):.3f} | baseline RMSE = {np.mean(bl):.3f}")

# 5b) the two rows of a match are not independent, so also split by match
gkf = GroupKFold(n_splits=5)
grm = [np.sqrt(np.mean((y[te] - LinearRegression().fit(df.loc[tr, X_COLS], y[tr])
                        .predict(df.loc[te, X_COLS])) ** 2))
       for tr, te in gkf.split(df, y, groups=df.match_id)]
print(f"Match-grouped CV RMSE = {np.mean(grm):.3f}")

# 6) actual vs predicted plot
pred = cross_val_predict(LinearRegression(), df[X_COLS], y, cv=5)
plt.figure(figsize=(5, 5)); plt.scatter(pred, y, s=16, alpha=.5)
lim = [-1, 7.5]; plt.plot(lim, lim, "k--")
plt.xlabel("predicted goals (cross-validated)"); plt.ylabel("actual goals scored")
plt.tight_layout(); plt.savefig(FIG / "actual_vs_predicted.png", dpi=150); plt.close()
print("saved EDA csvs, model files and figures")

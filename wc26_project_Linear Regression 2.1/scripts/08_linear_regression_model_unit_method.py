"""
on our 104-match dataset.
  T1  sklearn multiple linear regression, 60/40 split, MAE / MSE / RMSE / normalised RMSE / R2, vs mean baseline
  T2  full regression equation
  T3  statsmodels OLS summary (R2, adjusted R2, p-values) + linearity scatterplots
  T4-6 log transform: raw squad value vs log squad value (before / after)
  T7-8 correlation heatmap + remove one variable of the most correlated pair, rebuild, compare
  T9  z-score standardisation        T10 Yeo-Johnson (Gaussian-like) transformation
Input : features/final_model_2_1_104_rows.csv (+ clean/R2_squad_value.csv for the raw squad value)
Output: features/model_2_1_teacher_results.csv, features/model_2_1_equation.txt, figures/T*.png
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "features"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
df = pd.read_csv(F / "final_model_2_1_104_rows.csv")
X_COLS = ["fifa_points_diff", "log_market_value_diff", "front3_rating_mean_diff", "goals_scored_pm_diff",
          "goals_conceded_pm_diff", "points_pm_diff", "momentum_diff", "host_diff"]
Y = "goal_diff"
NICE = {"fifa_points_diff": "FIFA points (A-B)", "log_market_value_diff": "log squad value (A-B)",
        "front3_rating_mean_diff": "front-three rating (A-B)", "goals_scored_pm_diff": "goals scored per match (A-B)",
        "goals_conceded_pm_diff": "goals conceded per match (A-B)", "points_pm_diff": "points per match (A-B)",
        "momentum_diff": "tournament momentum (A-B)", "host_diff": "host advantage (A-B)"}
results = []


def run_sklearn(name, cols, data):
    """T1 steps: split 60/40, fit, predict, metrics; returns metrics dict."""
    x, y = data[cols].values, data[Y].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    out = dict(model=name, n_variables=len(cols), MAE=metrics.mean_absolute_error(y_test, y_pred),
               MSE=metrics.mean_squared_error(y_test, y_pred), RMSE=rmse,
               RMSE_normalised=rmse / (y.max() - y.min()), R2_test=metrics.r2_score(y_test, y_pred))
    ols = sm.OLS(y_train, sm.add_constant(pd.DataFrame(X_train, columns=cols))).fit()
    out["R2_train"] = ols.rsquared; out["adj_R2_train"] = ols.rsquared_adj
    return out, model, ols, (y_test, y_pred)


# ---------- T1: baseline vs linear regression ----------
r1, model, ols1, (y_test, y_pred) = run_sklearn("Linear regression (8 variables)", X_COLS, df)
X_train, X_test, y_train, _ = train_test_split(df[X_COLS].values, df[Y].values, test_size=0.4, random_state=0)
yb = np.full(len(y_test), y_train.mean())
base = dict(model="Baseline (predict training mean)", n_variables=0, MAE=metrics.mean_absolute_error(y_test, yb),
            MSE=metrics.mean_squared_error(y_test, yb), RMSE=math.sqrt(metrics.mean_squared_error(y_test, yb)))
base["RMSE_normalised"] = base["RMSE"] / (df[Y].max() - df[Y].min()); base["R2_test"] = metrics.r2_score(y_test, yb)
results += [base, r1]
print("T1  train rows:", len(X_train), "test rows:", len(X_test))
print(pd.DataFrame([base, r1])[["model", "MAE", "MSE", "RMSE", "RMSE_normalised", "R2_test"]].round(3).to_string(index=False))

# ---------- T2: equation ----------
eq = f"goal_diff = {model.intercept_:.4f} " + " ".join(f"{'+' if c >= 0 else '-'} {abs(c):.4f} ({v})" for v, c in zip(X_COLS, model.coef_))
(F / "model_2_1_equation.txt").write_text(eq, encoding="utf-8")
print("\nT2  equation:\n", eq)

# ---------- T3: statsmodels on the same training set + linearity scatterplots ----------
print("\nT3  statsmodels OLS summary (training set):\n", ols1.summary().tables[1])
print(f"    R2 = {ols1.rsquared:.3f} | adjusted R2 = {ols1.rsquared_adj:.3f}")
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for a, c in zip(ax.ravel(), X_COLS):
    a.scatter(df[c], df[Y], s=12); a.set_xlabel(NICE[c]); a.set_ylabel("goal difference (A-B)")
plt.tight_layout(); plt.savefig(FIG / "T3_linearity_scatterplots.png", dpi=150); plt.close()

# ---------- T4-T6: log transformation (raw squad value vs log squad value) ----------
r2 = pd.read_csv(ROOT / "clean" / "R2_squad_value.csv").set_index("team")
df["raw_market_value_diff_m"] = (df.team_a.map(r2.market_value_eur) - df.team_b.map(r2.market_value_eur)) / 1e6
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, c, t in zip(ax, ["raw_market_value_diff_m", "log_market_value_diff"], ["Original squad value difference (EUR m)", "Log squad value difference"]):
    a.scatter(df[c], df[Y], s=14); b = np.polyfit(df[c], df[Y], 1); xs = np.linspace(df[c].min(), df[c].max(), 50)
    a.plot(xs, np.polyval(b, xs)); a.set_title(f"{t}  (r = {df[c].corr(df[Y]):.2f})"); a.set_ylabel("goal difference")
plt.tight_layout(); plt.savefig(FIG / "T4_log_transform_before_after.png", dpi=150); plt.close()
cols_raw = [c if c != "log_market_value_diff" else "raw_market_value_diff_m" for c in X_COLS]
r_raw, *_ = run_sklearn("Using RAW squad value (no log)", cols_raw, df)
results.append(r_raw)
print(f"\nT4-6 correlation with goal diff: raw value {df.raw_market_value_diff_m.corr(df[Y]):.3f} -> log value {df.log_market_value_diff.corr(df[Y]):.3f}")
print(f"     test R2 raw = {r_raw['R2_test']:.3f} | test R2 log = {r1['R2_test']:.3f}")

# ---------- T7-T8: correlation heatmap and collinearity removal ----------
corr = df[X_COLS + [Y]].corr()
import seaborn as sns
plt.figure(figsize=(10, 8))
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=False, annot=True, fmt=".2f", annot_kws={"size": 7})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.tight_layout(); plt.savefig(FIG / "T7_correlation_heatmap.png", dpi=150); plt.close()
xc = df[X_COLS].corr().abs().where(~np.eye(len(X_COLS), dtype=bool))
a_, b_ = xc.stack().idxmax()
print(f"\nT7  most correlated pair of explanatory variables: {a_} & {b_}  (r = {df[a_].corr(df[b_]):.2f})")
drop = a_ if abs(df[a_].corr(df[Y])) < abs(df[b_].corr(df[Y])) else b_      # keep the one more related to goal difference
cols_nc = [c for c in X_COLS if c != drop]
r_nc, _, ols_nc, _ = run_sklearn(f"Collinearity step: removed {drop}", cols_nc, df)
results.append(r_nc)
print(f"T8  removed {drop}: adj R2 (train) {ols1.rsquared_adj:.3f} -> {ols_nc.rsquared_adj:.3f} | test R2 {r1['R2_test']:.3f} -> {r_nc['R2_test']:.3f}")

# ---------- T9: z-score standardisation ----------
dz = df.copy(); dz[X_COLS] = StandardScaler().fit_transform(df[X_COLS])
r_z, _, ols_z, _ = run_sklearn("z-score standardised", X_COLS, dz)
results.append(r_z)
print(f"\nT9  standardised: test R2 {r_z['R2_test']:.3f} (same fit as unscaled; only the coefficients become comparable)")
print("    standardised coefficients (effect of +1 SD):\n", ols_z.params.drop("const").rename(index=NICE).round(3).to_string())

# ---------- T10: Yeo-Johnson ----------
dy = df.copy(); dy[X_COLS] = PowerTransformer(method="yeo-johnson").fit_transform(df[X_COLS])
r_y, *_ = run_sklearn("Yeo-Johnson transformed", X_COLS, dy)
results.append(r_y)
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); v = "momentum_diff"
ax[0].hist(df[v], bins=12, density=True); ax[0].set_title(f"{v} (original)")
ax[1].hist(dy[v], bins=12, density=True); ax[1].set_title(f"{v} (Yeo-Johnson)")
plt.tight_layout(); plt.savefig(FIG / "T10_yeo_johnson.png", dpi=150); plt.close()
print(f"\nT10 Yeo-Johnson: test R2 {r_y['R2_test']:.3f} vs original {r1['R2_test']:.3f}")

# ---------- stability: the single 60/40 split has only 42 test matches, so repeat it 200 times ----------
rows = []
for seed in range(200):
    Xtr, Xte, ytr, yte = train_test_split(df[X_COLS].values, df[Y].values, test_size=0.4, random_state=seed)
    p = LinearRegression().fit(Xtr, ytr).predict(Xte)
    rows.append((math.sqrt(metrics.mean_squared_error(yte, p)), metrics.r2_score(yte, p), math.sqrt(metrics.mean_squared_error(yte, np.full(len(yte), ytr.mean())))))
s = np.array(rows).mean(0)
print(f"\nStability (200 random 60/40 splits): mean RMSE {s[0]:.2f} vs baseline {s[2]:.2f} | mean test R2 {s[1]:.2f}")

# ---------- statsmodels on the WHOLE dataset (exactly like the unit's scripts): summary + comparison ----------
def sm_full(cols, data):
    return sm.OLS(data[Y], sm.add_constant(data[cols])).fit()

full = sm_full(X_COLS, df)
print("\n=== statsmodels summary, all 104 matches (as in the unit scripts) ===\n", full.summary())
variants = {"1 original (8 variables)": sm_full(X_COLS, df),
            "2 raw squad value instead of log": sm_full(cols_raw, df),
            f"3 collinearity: removed {drop}": sm_full(cols_nc, df),
            "4 z-score standardised": sm_full(X_COLS, dz),
            "5 Yeo-Johnson": sm_full(X_COLS, dy)}
cmp = pd.DataFrame([{"model": k, "R2": m.rsquared, "adj_R2": m.rsquared_adj, "F_pvalue": m.f_pvalue, "AIC": m.aic} for k, m in variants.items()]).round(4)
cmp.to_csv(F / "model_2_1_statsmodels_comparison.csv", index=False)
print("\nComparison of the model versions (statsmodels, all 104 matches):\n", cmp.to_string(index=False))
(F / "model_2_1_statsmodels_summary.txt").write_text(str(full.summary()), encoding="utf-8")

res = pd.DataFrame(results).round(4); res.to_csv(F / "model_2_1_teacher_results.csv", index=False)
plt.figure(figsize=(5.5, 5.5)); plt.scatter(y_pred, y_test, s=18); plt.plot([-6, 6], [-6, 6], "k--")
plt.xlabel("predicted goal difference (test set)"); plt.ylabel("actual goal difference"); plt.tight_layout()
plt.savefig(FIG / "T1_actual_vs_predicted_test.png", dpi=150); plt.close()
print("\nsaved features/model_2_1_results.csv, features/model_2_1_equation.txt and figures/T*.png")

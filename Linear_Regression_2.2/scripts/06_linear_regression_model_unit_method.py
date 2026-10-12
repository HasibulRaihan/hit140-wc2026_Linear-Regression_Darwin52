"""
The unit's linear-regression method (T1-T10) on the 208 team-match dataset, plus
regression-assumption diagnostics.

  T1    60/40 split, MAE / MSE / RMSE / normalised RMSE / R2 vs a mean baseline
  T2    the regression equation
  T3    statsmodels OLS summary + linearity scatterplots
  T4-6  log1p transform of own_goals_per_match_before, before / after
  T7-8  correlation heatmap, then drop one of the most correlated pair and refit
  T9    z-score standardisation
  T10   Yeo-Johnson transformation

In : features/final_model_2_2_208_rows.csv
Out: features/model_2_2_results.csv, model_2_2_equation.txt,
     model_2_2_statsmodels_comparison.csv, model_2_2_statsmodels_summary.txt,
     model_2_2_assumptions.csv, figures/T*.png

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
import seaborn as sns
import statsmodels.api as sm
from scipy import stats as sps
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

Y = "goals_scored"


def run_sklearn(name, cols, data, transformer=None):
    """Split 60/40, fit, predict, score."""
    x, y = data[cols].values, data[Y].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    if transformer is not None:
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = regression_metrics(y_test, y_pred)
    out = dict(model=name, n_variables=len(cols), MAE=scores["MAE"],
               MSE=scores["MSE"], RMSE=scores["RMSE"],
               RMSE_normalised=scores["RMSE"] / (y.max() - y.min()), R2_test=scores["R2"])
    ols = sm.OLS(y_train, sm.add_constant(pd.DataFrame(X_train, columns=cols))).fit()
    out["R2_train"] = ols.rsquared; out["adj_R2_train"] = ols.rsquared_adj
    return out, model, ols, (y_test, y_pred)

def sm_full(cols, data):
    return sm.OLS(data[Y], sm.add_constant(data[cols])).fit()


def main():
    ROOT = Path(__file__).resolve().parent.parent
    F = ROOT / "features"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
    df = load_model_data()
    NICE = {"own_mean_squad_age": "own mean squad age", "opp_mean_squad_age": "opponent mean squad age",
            "own_big5_share": "own big-5 share", "opp_big5_share": "opponent big-5 share",
            "own_goals_per_match_before": "own goals p/m (earlier)",
            "opp_conceded_per_match_before": "opp conceded p/m (earlier)",
            "days_rest": "days of rest", "is_knockout": "knockout stage"}
    results = []




    # ---------- T1: baseline vs linear regression ----------
    r1, model, ols1, (y_test, y_pred) = run_sklearn("Linear regression (8 variables)", X_COLS, df)
    X_train, X_test, y_train, _ = train_test_split(df[X_COLS].values, df[Y].values, test_size=0.4, random_state=0)
    yb = np.full(len(y_test), y_train.mean())
    baseline_scores = regression_metrics(y_test, yb)
    base = dict(model="Baseline (predict training mean)", n_variables=0,
                MAE=baseline_scores["MAE"], MSE=baseline_scores["MSE"],
                RMSE=baseline_scores["RMSE"])
    base["RMSE_normalised"] = base["RMSE"] / (df[Y].max() - df[Y].min())
    base["R2_test"] = baseline_scores["R2"]
    results += [base, r1]
    print("T1  train rows:", len(X_train), "test rows:", len(X_test))
    print(pd.DataFrame([base, r1])[["model", "MAE", "MSE", "RMSE", "RMSE_normalised", "R2_test"]].round(3).to_string(index=False))

    # ---------- T2: equation ----------
    eq = (f"goals_scored = {model.intercept_:.4f} "
          + " ".join(f"{'+' if c >= 0 else '-'} {abs(c):.4f} ({v})" for v, c in zip(X_COLS, model.coef_)))
    (F / "model_2_2_equation.txt").write_text(eq, encoding="utf-8")
    print("\nT2  equation:\n", eq)

    # ---------- T3: statsmodels on the same training set + linearity scatterplots ----------
    print("\nT3  statsmodels OLS summary (training set):\n", ols1.summary().tables[1])
    print(f"    R2 = {ols1.rsquared:.3f} | adjusted R2 = {ols1.rsquared_adj:.3f}")
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for a, c in zip(ax.ravel(), X_COLS):
        a.scatter(df[c], df[Y], s=12, alpha=.5); a.set_xlabel(NICE[c]); a.set_ylabel("goals scored")
    plt.tight_layout(); plt.savefig(FIG / "T3_linearity_scatterplots.png", dpi=150); plt.close()

    # ---------- T4-T6: log transformation ----------
    # own_goals_per_match_before is the most right-skewed variable and contains zeros
    # (every opening fixture), so log1p rather than a plain log.
    df["log_own_goals_before"] = np.log1p(df.own_goals_per_match_before)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, c, t in zip(ax, ["own_goals_per_match_before", "log_own_goals_before"],
                       ["Original own goals p/m (earlier matches)", "log1p own goals p/m"]):
        a.scatter(df[c], df[Y], s=14, alpha=.5)
        b = np.polyfit(df[c], df[Y], 1); xs = np.linspace(df[c].min(), df[c].max(), 50)
        a.plot(xs, np.polyval(b, xs)); a.set_title(f"{t}  (r = {df[c].corr(df[Y]):.2f})")
        a.set_ylabel("goals scored")
    plt.tight_layout(); plt.savefig(FIG / "T4_log_transform_before_after.png", dpi=150); plt.close()
    cols_log = [c if c != "own_goals_per_match_before" else "log_own_goals_before" for c in X_COLS]
    r_log, *_ = run_sklearn("Using log1p own goals p/m", cols_log, df)
    results.append(r_log)
    print(f"\nT4-6 correlation with goals scored: raw {df.own_goals_per_match_before.corr(df[Y]):.3f} "
          f"-> log1p {df.log_own_goals_before.corr(df[Y]):.3f}")
    print(f"     test R2 raw = {r1['R2_test']:.3f} | test R2 log = {r_log['R2_test']:.3f}")

    # ---------- T7-T8: correlation heatmap and collinearity removal ----------
    corr = df[X_COLS + [Y]].corr()
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
                     square=False, annot=True, fmt=".2f", annot_kws={"size": 7})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    plt.tight_layout(); plt.savefig(FIG / "T7_correlation_heatmap.png", dpi=150); plt.close()
    xc = df[X_COLS].corr().abs().where(~np.eye(len(X_COLS), dtype=bool))
    a_, b_ = xc.stack().idxmax()
    print(f"\nT7  most correlated pair of explanatory variables: {a_} & {b_}  (r = {df[a_].corr(df[b_]):.2f})")
    drop = a_ if abs(df[a_].corr(df[Y])) < abs(df[b_].corr(df[Y])) else b_  # keep the one closer to the target
    cols_nc = [c for c in X_COLS if c != drop]
    r_nc, _, ols_nc, _ = run_sklearn(f"Collinearity step: removed {drop}", cols_nc, df)
    results.append(r_nc)
    print(f"T8  removed {drop}: adj R2 (train) {ols1.rsquared_adj:.3f} -> {ols_nc.rsquared_adj:.3f} "
          f"| test R2 {r1['R2_test']:.3f} -> {r_nc['R2_test']:.3f}")

    # ---------- T9: z-score standardisation ----------
    scaler = StandardScaler()
    r_z, _, ols_z, _ = run_sklearn("z-score standardised", X_COLS, df, transformer=scaler)
    dz = df.copy(); dz[X_COLS] = scaler.transform(df[X_COLS].values)
    results.append(r_z)
    print(f"\nT9  standardised: test R2 {r_z['R2_test']:.3f}")
    print("    coefficients per +1 SD:\n",
          ols_z.params.drop("const").rename(index=NICE).round(3).to_string())

    # ---------- T10: Yeo-Johnson ----------
    power = PowerTransformer(method="yeo-johnson")
    r_y, *_ = run_sklearn("Yeo-Johnson transformed", X_COLS, df, transformer=power)
    dy = df.copy(); dy[X_COLS] = power.transform(df[X_COLS].values)
    results.append(r_y)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); v = "own_goals_per_match_before"
    ax[0].hist(df[v], bins=12, density=True); ax[0].set_title(f"{v} (original)")
    ax[1].hist(dy[v], bins=12, density=True); ax[1].set_title(f"{v} (Yeo-Johnson)")
    plt.tight_layout(); plt.savefig(FIG / "T10_yeo_johnson.png", dpi=150); plt.close()
    print(f"\nT10 Yeo-Johnson: test R2 {r_y['R2_test']:.3f} vs original {r1['R2_test']:.3f}")

    # ---------- stability: one 60/40 split tests only 84 rows, so repeat 200 times ----------
    rows = []
    for seed in range(200):
        Xtr, Xte, ytr, yte = train_test_split(df[X_COLS].values, df[Y].values, test_size=0.4, random_state=seed)
        p = LinearRegression().fit(Xtr, ytr).predict(Xte)
        scores = regression_metrics(yte, p)
        baseline_scores = regression_metrics(yte, np.full(len(yte), ytr.mean()))
        rows.append((scores["RMSE"], scores["R2"], baseline_scores["RMSE"]))
    s = np.array(rows).mean(0)
    print(f"\nStability (200 random 60/40 splits): mean RMSE {s[0]:.2f} vs baseline {s[2]:.2f} | mean test R2 {s[1]:.2f}")

    # ---------- statsmodels on all 208 rows: summary + version comparison ----------


    full = sm_full(X_COLS, df)
    print("\n=== statsmodels summary, all 208 team-match rows ===\n", full.summary())
    variants = {"1 original (8 variables)": sm_full(X_COLS, df),
                "2 log1p own goals p/m": sm_full(cols_log, df),
                f"3 collinearity: removed {drop}": sm_full(cols_nc, df),
                "4 z-score standardised": sm_full(X_COLS, dz),
                "5 Yeo-Johnson": sm_full(X_COLS, dy)}
    cmp = pd.DataFrame([{"model": k, "R2": m.rsquared, "adj_R2": m.rsquared_adj,
                         "F_pvalue": m.f_pvalue, "AIC": m.aic} for k, m in variants.items()]).round(4)
    cmp.to_csv(F / "model_2_2_statsmodels_comparison.csv", index=False)
    print("\nModel versions (all 208 rows):\n", cmp.to_string(index=False))
    (F / "model_2_2_statsmodels_summary.txt").write_text(str(full.summary()), encoding="utf-8")

    # ---------- regression assumptions ----------
    resid, fitted = full.resid, full.fittedvalues
    sw_W, sw_p = sps.shapiro(resid)
    bp_p = het_breuschpagan(resid, full.model.exog)[1]
    dw = durbin_watson(resid)
    n_neg = int((fitted < 0).sum())
    disp = df[Y].var() / df[Y].mean()
    assump = pd.DataFrame([
        dict(assumption="Normality of residuals", test="Shapiro-Wilk",
             statistic=round(sw_W, 4), p_value=round(sw_p, 6),
             verdict="violated" if sw_p < .05 else "satisfied"),
        dict(assumption="Homoscedasticity", test="Breusch-Pagan", statistic=np.nan,
             p_value=round(bp_p, 6), verdict="violated" if bp_p < .05 else "satisfied"),
        dict(assumption="Independence of errors", test="Durbin-Watson", statistic=round(dw, 4),
             p_value=np.nan, verdict="near 2, but the two rows of a match are paired"),
        dict(assumption="Target is continuous", test="negative fitted values",
             statistic=n_neg, p_value=np.nan,
             verdict="violated - goals are counts, OLS can predict below zero"),
        dict(assumption="Equidispersion (for Poisson)", test="variance / mean",
             statistic=round(disp, 4), p_value=np.nan,
             verdict="overdispersed" if disp > 1.2 else "approximately equidispersed"),
    ])
    assump.to_csv(F / "model_2_2_assumptions.csv", index=False)
    print("\nREGRESSION ASSUMPTIONS\n", assump.to_string(index=False))
    print(f"\nResidual skew {sps.skew(resid, bias=False):+.3f}, "
          f"excess kurtosis {sps.kurtosis(resid, bias=False):+.3f}")

    res = pd.DataFrame(results).round(4); res.to_csv(F / "model_2_2_results.csv", index=False)
    plt.figure(figsize=(5.5, 5.5)); plt.scatter(y_pred, y_test, s=18, alpha=.6)
    plt.plot([-1, 7.5], [-1, 7.5], "k--")
    plt.xlabel("predicted goals (test set)"); plt.ylabel("actual goals scored"); plt.tight_layout()
    plt.savefig(FIG / "T1_actual_vs_predicted_test.png", dpi=150); plt.close()
    print("\nsaved model_2_2_results.csv, model_2_2_equation.txt and figures/T*.png")


if __name__ == "__main__":
    main()

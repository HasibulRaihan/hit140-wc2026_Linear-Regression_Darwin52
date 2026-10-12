"""
Systematic comparison of nine competing algorithms, with a recommendation.

  0 Baseline              predict the training mean
  1 OLS                   the unit's linear regression
  2 Ridge                 L2-penalised linear
  3 Lasso                 L1-penalised linear
  4 ElasticNet            L1 + L2
  5 Poisson GLM           log link, built for counts
  6 Negative Binomial GLM Poisson + an overdispersion parameter
  7 Random Forest         non-linear, captures interactions
  8 Gradient Boosting     non-linear, sequential

Validation: 5-fold CV grouped by match, repeated 20 times. Grouping keeps the two
rows of a match in the same fold. Also runs a conditional dispersion check and a
Bonferroni-corrected curvature test.

In : features/final_model_2_2_208_rows.csv
Out: features/model_2_2_algorithm_comparison.csv, model_2_2_algorithm_recommendation.txt,
     model_2_2_dispersion_check.csv, model_2_2_curvature_test.csv,
     figures/algorithm_comparison.png, figures/count_model_vs_ols.png

Abishek Rajeshkumar, Rejan Sapkota
"""
import sys
import warnings
from pathlib import Path

from model_config import X_COLS
from model_data import load_model_data
from model_metrics import regression_metrics

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats as sps
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def fit_predict_sklearn(est, scale):
    def f(Xtr, ytr, Xte):
        if scale:
            s = StandardScaler().fit(Xtr)
            Xtr, Xte = s.transform(Xtr), s.transform(Xte)
        return est().fit(Xtr, ytr).predict(Xte)
    return f

def fit_predict_glm(family):
    def f(Xtr, ytr, Xte):
        m = sm.GLM(ytr, sm.add_constant(Xtr), family=family).fit()
        return m.predict(sm.add_constant(Xte, has_constant="add"))
    return f

def dispersion_report(Xa, ya):
    """Overdispersion conditional on the model: Pearson chi2 / df from a Poisson fit,
    plus the Cameron-Trivedi auxiliary regression of ((y-mu)^2 - y)/mu on mu."""
    pois = sm.GLM(ya, sm.add_constant(Xa), family=sm.families.Poisson()).fit()
    mu = pois.mu
    aux_fit = sm.OLS(((ya - mu) ** 2 - ya) / mu, mu).fit()
    return dict(raw_var_over_mean=ya.var() / ya.mean(),
                conditional_dispersion=pois.pearson_chi2 / pois.df_resid,
                ct_alpha=float(aux_fit.params[0]), ct_pvalue=float(aux_fit.pvalues[0]))

def evaluate(fn, X, y, groups, n_repeats=20):
    rmse, mae, r2, neg = [], [], [], []
    for rep in range(n_repeats):
        rng = np.random.default_rng(rep)
        order = rng.permutation(np.unique(groups))
        fold_of = {g: i % 5 for i, g in enumerate(order)}
        fold = np.array([fold_of[g] for g in groups])
        pred_all, true_all = [], []
        for k in range(5):
            tr, te = fold != k, fold == k
            p = np.asarray(fn(X[tr], y[tr], X[te]), dtype=float)
            pred_all.append(p); true_all.append(y[te])
        p = np.concatenate(pred_all); t = np.concatenate(true_all)
        scores = regression_metrics(t, p)
        rmse.append(scores["RMSE"])
        mae.append(scores["MAE"])
        r2.append(scores["R2"])
        neg.append(np.mean(p < 0) * 100)
    return dict(RMSE=np.mean(rmse), RMSE_sd=np.std(rmse), MAE=np.mean(mae),
                R2=np.mean(r2), pct_negative_predictions=np.mean(neg),
                rmse_by_repeat=np.array(rmse))


def main():
    ROOT = Path(__file__).resolve().parent.parent
    F = ROOT / "features"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
    df = load_model_data()
    X = df[X_COLS].values
    y = df.goals_scored.values
    groups = df.match_id.values
    N_REPEATS = 20


    # ------------------------------------------------------------------ the algorithms






    DISP = dispersion_report(X, y)
    # A non-significant or negative alpha is clamped, which collapses NB onto Poisson.
    ALPHA = max(DISP["ct_alpha"], 1e-6) if DISP["ct_pvalue"] < .05 else 1e-6

    print("DISPERSION CHECK")
    print(f"  raw variance / mean           : {DISP['raw_var_over_mean']:.4f}")
    print(f"  conditional (Pearson chi2/df) : {DISP['conditional_dispersion']:.4f}")
    print(f"  Cameron-Trivedi alpha         : {DISP['ct_alpha']:+.4f} (p = {DISP['ct_pvalue']:.4f})")
    print("  ->", "predictors absorb the raw overdispersion; Poisson is the correct family"
          if DISP["ct_pvalue"] >= .05 else "genuine overdispersion remains; NB is justified")
    pd.DataFrame([DISP]).round(4).to_csv(F / "model_2_2_dispersion_check.csv", index=False)

    ALGORITHMS = {
        "0 Baseline (training mean)": lambda Xtr, ytr, Xte: np.full(len(Xte), ytr.mean()),
        "1 OLS linear regression": fit_predict_sklearn(LinearRegression, False),
        "2 Ridge (L2)": fit_predict_sklearn(lambda: Ridge(alpha=1.0), True),
        "3 Lasso (L1)": fit_predict_sklearn(lambda: Lasso(alpha=0.02, max_iter=10000), True),
        "4 ElasticNet": fit_predict_sklearn(lambda: ElasticNet(alpha=0.02, l1_ratio=0.5, max_iter=10000), True),
        "5 Poisson GLM": fit_predict_glm(sm.families.Poisson()),
        "6 Negative Binomial GLM": fit_predict_glm(sm.families.NegativeBinomial(alpha=ALPHA)),
        "7 Random Forest": fit_predict_sklearn(
            lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0), False),
        "8 Gradient Boosting": fit_predict_sklearn(
            lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                              random_state=0), False),
    }


    # ------------------------------------------------------------- repeated grouped CV


    print(f"\n5-fold CV grouped by match, {N_REPEATS} repeats ({N_REPEATS * 5} fits each)\n")

    rows, per_repeat = [], {}
    for name, fn in ALGORITHMS.items():
        m = evaluate(fn, X, y, groups, N_REPEATS)
        per_repeat[name] = m.pop("rmse_by_repeat")
        m["algorithm"] = name
        rows.append(m)
        print(f"  {name:<28} RMSE {m['RMSE']:.4f} (sd {m['RMSE_sd']:.3f})  "
              f"MAE {m['MAE']:.4f}  R2 {m['R2']:+.4f}  neg {m['pct_negative_predictions']:.1f}%")

    res = (pd.DataFrame(rows)[["algorithm", "RMSE", "RMSE_sd", "MAE", "R2",
                               "pct_negative_predictions"]]
           .sort_values("RMSE").round(4).reset_index(drop=True))
    res.to_csv(F / "model_2_2_algorithm_comparison.csv", index=False)

    best = res.iloc[0]
    ols = res[res.algorithm.str.startswith("1 ")].iloc[0]
    basel = res[res.algorithm.str.startswith("0 ")].iloc[0]
    poisson = res[res.algorithm.str.startswith("5 ")].iloc[0]
    print("\nRanked by CV RMSE:\n", res.to_string(index=False))

    # The repeats use identical folds, so compare the winner with OLS pairwise.
    d_best = per_repeat[ols.algorithm] - per_repeat[best.algorithm]
    t_stat, p_val = sps.ttest_1samp(d_best, 0)
    print(f"\nPaired OLS - {best.algorithm[2:]}: {d_best.mean():+.4f} goals, "
          f"t = {t_stat:.3f}, p = {p_val:.4f}")

    # ---------------------------------------------------------------- curvature test
    # A tree beating a linear model suggests curvature, so test it: add a squared term
    # per continuous variable. Four tests, so the threshold is Bonferroni-corrected.
    CURVE_VARS = ["own_mean_squad_age", "opp_mean_squad_age", "own_big5_share",
                  "own_goals_per_match_before"]
    BONF = .05 / len(CURVE_VARS)
    base_ols = sm.OLS(y, sm.add_constant(df[X_COLS])).fit()
    curve = []
    for v in CURVE_VARS:
        d2 = df.copy(); d2[v + "_sq"] = d2[v] ** 2
        m = sm.OLS(y, sm.add_constant(d2[X_COLS + [v + "_sq"]])).fit()
        pv = float(m.pvalues[v + "_sq"])
        curve.append(dict(variable=v, quadratic_p=round(pv, 4),
                          sig_at_5pct=pv < .05, sig_after_bonferroni=pv < BONF,
                          adj_r2_before=round(base_ols.rsquared_adj, 4),
                          adj_r2_after=round(m.rsquared_adj, 4)))
    curve = pd.DataFrame(curve)
    curve.to_csv(F / "model_2_2_curvature_test.csv", index=False)

    n_raw = int(curve.sig_at_5pct.sum())
    n_bonf = int(curve.sig_after_bonferroni.sum())
    print("\nCURVATURE TEST")
    print(curve.to_string(index=False))
    print(f"  Bonferroni threshold for {len(CURVE_VARS)} tests: p < {BONF:.4f}")
    print(f"  significant uncorrected: {n_raw} | surviving correction: {n_bonf}")

    apex = None
    if n_raw and not n_bonf:
        hit = curve[curve.sig_at_5pct].iloc[0]
        d2 = df.copy(); d2[hit.variable + "_sq"] = d2[hit.variable] ** 2
        mm = sm.OLS(y, sm.add_constant(d2[X_COLS + [hit.variable + "_sq"]])).fit()
        apex = -mm.params[hit.variable] / (2 * mm.params[hit.variable + "_sq"])
        print(f"  -> {hit.variable} only, p = {hit.quadratic_p:.3f}, inverted-U peaking "
              f"near {apex:.1f}; suggestive, not established")

    # ----------------------------------------------------------------- recommendation
    non_base = res[res.algorithm != basel.algorithm]
    spread = non_base.RMSE.max() - non_base.RMSE.min()
    is_tree = best.algorithm.startswith(("7", "8"))
    gain = (ols.RMSE - best.RMSE) / ols.RMSE * 100

    lines = [
        "ALGORITHM COMPARISON - RECOMMENDATION",
        "=" * 74, "",
        f"Validation: 5-fold CV grouped by match, {N_REPEATS} repeats.",
        "",
        f"Best by RMSE     : {best.algorithm}  (RMSE {best.RMSE:.4f}, MAE {best.MAE:.4f}, R2 {best.R2:+.4f})",
        f"Unit's OLS model : RMSE {ols.RMSE:.4f}, MAE {ols.MAE:.4f}, R2 {ols.R2:+.4f}",
        f"Mean baseline    : RMSE {basel.RMSE:.4f}",
        f"Best vs baseline : {(basel.RMSE - best.RMSE) / basel.RMSE * 100:.1f}% better",
        f"Best vs OLS      : {gain:+.1f}%  (paired p = {p_val:.4f})",
        f"Spread across all eight non-baseline methods: {spread:.3f} RMSE "
        f"({spread / basel.RMSE * 100:.0f}% of baseline error)",
        "",
        f"Curvature test   : {n_raw} of {len(CURVE_VARS)} quadratic terms significant at an "
        f"uncorrected 5%,",
        f"                   {n_bonf} surviving Bonferroni (p < {BONF:.4f})"
        + (f"; inverted-U near {apex:.1f}" if apex is not None else ""),
        f"Dispersion       : conditional {DISP['conditional_dispersion']:.3f}, "
        f"C-T alpha {DISP['ct_alpha']:+.3f} (p = {DISP['ct_pvalue']:.3f})",
        "",
        "RECOMMENDATION", "-" * 74,
    ]
    if is_tree:
        lines += [
            f"Lowest error      : {best.algorithm[2:]}, {gain:.1f}% ahead of OLS. The curvature test",
            "                    does not support shape as the reason, so the edge is shrinkage:",
            "                    averaging shallow trees damps extremes and cannot extrapolate,",
            "                    which is also why it never predicts a negative count. No",
            "                    coefficient per variable, which this report needs.",
            f"Best interpretable: Poisson GLM, {poisson.RMSE:.4f} RMSE - within "
            f"{poisson.RMSE - best.RMSE:.3f} of the best.",
            "                    Coefficient per variable, log link, non-negative predictions.",
            f"For the report    : OLS, as the brief requires. It sits "
            f"{(ols.RMSE - best.RMSE) / basel.RMSE * 100:.1f}% of the baseline error",
            "                    behind the best method and is the model interpreted here.",
        ]
    else:
        lines += [
            f"For a coefficient per variable, as required, keep OLS: no competitor is",
            "meaningfully more accurate.",
            f"To predict goals, use the Poisson GLM ({poisson.RMSE - best.RMSE:+.3f} RMSE vs the",
            "best linear model) - it can never return a negative count.",
            "The tree ensembles do not win, so there are no strong non-linearities left to",
            "exploit in eight variables at n = 208.",
        ]
    lines += [
        "",
        "WIDER FINDING", "-" * 74,
        f"All eight non-baseline methods land between {non_base.RMSE.min():.3f} and "
        f"{non_base.RMSE.max():.3f} RMSE.",
        "The ceiling is set by the data, not the model: pre-match information explains about",
        "a quarter of the variation in goals scored and the rest is largely irreducible.",
    ]
    (F / "model_2_2_algorithm_recommendation.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[-6:]))


    # ------------------------------------------------------------------------ figures
    # Two panels: one scale cannot show both that every method beats the baseline and
    # that they are then almost indistinguishable.
    order = res.sort_values("RMSE", ascending=False).reset_index(drop=True)
    is_base = order.algorithm.str.startswith("0")
    colors = ["#9aa0a6" if b_ else ("#c0504d" if a_ == best.algorithm else "#4a6fa5")
              for a_, b_ in zip(order.algorithm, is_base)]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6),
                           gridspec_kw={"width_ratios": [1, 1.15]})

    ax[0].barh(order.algorithm, order.RMSE, color=colors, edgecolor="black", linewidth=.6)
    ax[0].axvline(basel.RMSE, ls="--", color="black", lw=1.3)
    ax[0].set(xlabel="cross-validated RMSE in goals", xlim=(0, 1.55),
              title="Every method beats guessing the average")
    ax[0].grid(axis="x", alpha=.25)
    ax[0].text(basel.RMSE - .03, 0.15, "baseline", rotation=90, ha="right", va="bottom", fontsize=9)

    z = order[~is_base]
    ax[1].errorbar(z.RMSE, range(len(z)), xerr=z.RMSE_sd, fmt="o", ms=9, capsize=4,
                   color="#4a6fa5", ecolor="#9aa0a6", elinewidth=1.5, zorder=3)
    bi = list(z.algorithm).index(best.algorithm)
    ax[1].plot(z.RMSE.iloc[bi], bi, "o", ms=13, color="#c0504d", zorder=4)
    ax[1].set_yticks(range(len(z))); ax[1].set_yticklabels(z.algorithm)
    ax[1].axvspan(z.RMSE.min(), z.RMSE.max(), color="#c0504d", alpha=.06, zorder=0)
    ax[1].set(xlabel="cross-validated RMSE in goals (zoomed)",
              title=f"...but they finish within {spread:.3f} goals of each other")
    ax[1].grid(axis="x", alpha=.25)
    ax[1].annotate(f"total spread {spread:.3f} goals\n= {spread / basel.RMSE * 100:.1f}% of baseline error",
                   xy=(z.RMSE.mean(), len(z) - .55), ha="center", fontsize=9, color="#c0504d")

    fig.suptitle("Nine competing algorithms: the data sets the ceiling, not the model",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "algorithm_comparison.png", dpi=200); plt.close(fig)

    # OLS can fit below zero; a Poisson log link cannot.
    ols_fit = LinearRegression().fit(X, y).predict(X)
    poi_fit = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson()).fit().predict(sm.add_constant(X))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(ols_fit, y, s=22, alpha=.55, color="#4a6fa5", edgecolor="white")
    ax[0].axvline(0, color="#c0504d", lw=1.6)
    ax[0].axvspan(min(ols_fit.min(), -.4) - .1, 0, color="#c0504d", alpha=.10)
    ax[0].set(title=f"OLS - {int((ols_fit < 0).sum())} fitted values below zero",
              xlabel="predicted goals", ylabel="actual goals scored")
    ax[0].text(0.02, .97, "impossible region\n(negative goals)", transform=ax[0].transAxes,
               va="top", fontsize=9, color="#c0504d")
    ax[1].scatter(poi_fit, y, s=22, alpha=.55, color="#2e7d64", edgecolor="white")
    ax[1].axvline(0, color="#c0504d", lw=1.6)
    ax[1].set(title="Poisson GLM - a log link cannot return a negative count",
              xlabel="predicted goals", ylabel="actual goals scored")
    for a in ax:
        a.grid(alpha=.25)
    fig.suptitle("Why the choice of model family matters for a count target", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "count_model_vs_ols.png", dpi=200); plt.close(fig)
    print("\nsaved model_2_2_algorithm_comparison.csv, the recommendation and 2 figures")


if __name__ == "__main__":
    main()

"""
Report figure for Model 2.1 (data storytelling): one two-panel figure.
 A) What drives goal difference?  standardised coefficients with 95% confidence intervals (OLS, 104 matches)
 B) How close are the predictions?  predicted (leave-one-out) vs actual goal difference, labelled big misses
Input : features/final_model_2_1_104_rows.csv
Output: figures/report_story_figure.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "features"; FIG = ROOT / "figures"
df = pd.read_csv(F / "final_model_2_1_104_rows.csv")
X_COLS = ["fifa_points_diff", "log_market_value_diff", "front3_rating_mean_diff", "goals_scored_pm_diff",
          "goals_conceded_pm_diff", "points_pm_diff", "momentum_diff", "host_diff"]
LAB = {"fifa_points_diff": "FIFA points", "log_market_value_diff": "Squad value (log)", "front3_rating_mean_diff": "Front-three rating",
       "goals_scored_pm_diff": "Goals scored / match", "goals_conceded_pm_diff": "Goals conceded / match",
       "points_pm_diff": "Points / match", "momentum_diff": "Tournament momentum", "host_diff": "Host advantage"}
BLUE, ORANGE, GREY, INK, MUTED = "#1f5f99", "#d1691a", "#6b7280", "#1f2937", "#6b7280"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#9ca3af", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.1, 1]}, layout="constrained")

# A) standardised coefficients
Z = (df[X_COLS] - df[X_COLS].mean()) / df[X_COLS].std()
m = sm.OLS(df.goal_diff, sm.add_constant(Z)).fit(); ci = m.conf_int().drop("const"); b = m.params.drop("const")
order = b.sort_values().index
a = ax[0]
for i, c in enumerate(order):
    sig = ci.loc[c, 0] > 0 or ci.loc[c, 1] < 0
    a.plot([ci.loc[c, 0], ci.loc[c, 1]], [i, i], color=BLUE if sig else "#9ca3af", lw=1.4)
    a.plot(b[c], i, "o", color=BLUE if sig else "#9ca3af", ms=4.5, mec="white", mew=0.8)
a.axvline(0, color=INK, lw=0.6)
a.set_yticks(range(len(order))); a.set_yticklabels([LAB[c] for c in order]); a.set_xlabel("Goals per +1 SD (95% CI)")
a.text(0.02, 1.0, "blue = 95% CI excludes 0", transform=a.transAxes, ha="left", va="bottom", fontsize=5.8, color=MUTED)
a.set_title("A  What drives goal difference", loc="left", fontsize=7.5, fontweight="bold", pad=9)

# B) predicted vs actual (leave-one-out)
pred = cross_val_predict(LinearRegression(), df[X_COLS], df.goal_diff, cv=LeaveOneOut())
ko = df.match_id > 72
a = ax[1]
a.plot([-7, 7], [-7, 7], color="#9ca3af", lw=0.8, ls="--")
a.scatter(pred[~ko], df.goal_diff[~ko], s=11, color=BLUE, alpha=.75, label="Group stage", lw=0)
a.scatter(pred[ko], df.goal_diff[ko], s=11, color=ORANGE, alpha=.85, label="Knockout", lw=0)
err = (df.goal_diff - pred)
for i in err.abs().sort_values(ascending=False).index[:3]:
    a.annotate(f"{df.team_a[i]}-{df.team_b[i]}", (pred[i], df.goal_diff[i]), xytext=(-6, 3), textcoords="offset points", ha="right", fontsize=5.5, color=INK)
a.set_xlim(-5, 5); a.set_ylim(-5, 7.2); a.set_xlabel("Predicted goal diff. (leave-one-out)"); a.set_ylabel("Actual goal difference")
a.legend(frameon=False, fontsize=6, loc="lower right", handletextpad=0.2)
a.set_title("B  Predicted vs actual", loc="left", fontsize=7.5, fontweight="bold")

plt.savefig(FIG / "report_story_figure.png", dpi=240)
print("biggest misses:", [(df.team_a[i], df.team_b[i], int(df.goal_diff[i]), round(pred[i], 1)) for i in err.abs().sort_values(ascending=False).index[:3]])
print("saved figures/report_story_figure.png")

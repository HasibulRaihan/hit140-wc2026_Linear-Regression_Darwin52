"""
Model 2.1 - does the train/test split size or the random seed matter?
Tries 60/40, 70/30 and 80/20 splits. Each is run with random_state = 0 and then with 50 different seeds (0-49).
Input : features/final_model_2_1_104_rows.csv
Output: features/split_comparison.csv  (used in the report table 1.3)
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "features" / "final_model_2_1_104_rows.csv")
X_COLS = ["fifa_points_diff", "log_market_value_diff", "front3_rating_mean_diff", "goals_scored_pm_diff",
          "goals_conceded_pm_diff", "points_pm_diff", "momentum_diff", "host_diff"]
x, y = df[X_COLS].values, df["goal_diff"].values

rows = []
for name, test_size in [("60/40", 0.4), ("70/30", 0.3), ("80/20", 0.2)]:
    r2s, rmses = [], []
    for seed in range(50):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
        pred = LinearRegression().fit(X_train, y_train).predict(X_test)
        r2s.append(metrics.r2_score(y_test, pred))
        rmses.append(math.sqrt(metrics.mean_squared_error(y_test, pred)))
    rows.append(dict(split=name, R2_seed0=r2s[0], RMSE_seed0=rmses[0], R2_mean_50_seeds=np.mean(r2s),
                     R2_sd_50_seeds=np.std(r2s), RMSE_mean_50_seeds=np.mean(rmses)))
out = pd.DataFrame(rows).round(3)
out.to_csv(ROOT / "features" / "split_comparison.csv", index=False)
print(out.to_string(index=False))

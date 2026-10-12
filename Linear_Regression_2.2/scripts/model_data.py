from pathlib import Path

import pandas as pd

from model_config import X_COLS


def load_model_data():
    path = Path(__file__).resolve().parent.parent / "features" / "final_model_2_2_208_rows.csv"
    data = pd.read_csv(path)
    required = ["match_id", "date", "team", "opponent", "goals_scored"] + X_COLS
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing model dataset columns: {', '.join(missing)}")
    return data

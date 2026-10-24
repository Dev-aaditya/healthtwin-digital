
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "glucose_history.csv"


def get_twin_state():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Glucose history not found. Run "
            "python data/generate_data.py first."
        )

    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    latest = df.iloc[-1]
    average_glucose = float(df["glucose_mg_dl"].mean())

    # Illustrative reporting range, not a personal medical target
    time_in_range = float(
        df["glucose_mg_dl"].between(70, 180).mean() * 100
    )

    recent = df["glucose_mg_dl"].tail(4).mean()
    previous = df["glucose_mg_dl"].tail(8).head(4).mean()

    if recent > previous + 2:
        trend = "rising"
    elif recent < previous - 2:
        trend = "falling"
    else:
        trend = "stable"

    return {
        "data_type": "synthetic_demo",
        "readings": len(df),
        "latest_timestamp": latest["timestamp"].isoformat(),
        "latest_glucose_mg_dl": float(latest["glucose_mg_dl"]),
        "average_glucose_mg_dl": round(average_glucose, 1),
        "time_in_range_percent": round(time_in_range, 1),
        "recent_trend": trend,
    }


from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "data" / "glucose_history.csv"


def generate_glucose_data(days=7, seed=42):
    rng = np.random.default_rng(seed)

    intervals = days * 24 * 4  # One reading every 15 minutes
    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("15min"),
        periods=intervals,
        freq="15min",
    )

    hours = timestamps.hour + timestamps.minute / 60

    # Simulated daily glucose rhythm
    glucose = (
        110
        + 8 * np.sin(2 * np.pi * (hours - 6) / 24)
        + rng.normal(0, 7, intervals)
    )
    glucose = np.asarray(glucose, dtype=float).copy()

    # Simulated meals and their glucose responses
    meal_carbs = np.zeros(intervals)
    meal_times = [8, 13, 19]

    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour + timestamp.minute / 60

        for meal_time in meal_times:
            elapsed = hour - meal_time

            if 0 <= elapsed <= 2:
                if abs(elapsed) < 0.25:
                    meal_carbs[i] = rng.choice([30, 45, 60])

                # Simplified post-meal glucose response
                glucose[i] += (
                    35
                    * np.exp(-elapsed / 1.1)
                    * (1 + 0.2 * np.sin(timestamp.dayofyear))
                )
                

    activity = rng.choice(
        [0, 0, 0, 15, 30],
        size=intervals,
    )

    sleep = np.clip(
        rng.normal(7, 0.8, intervals),
        4,
        9,
    )

    stress = rng.integers(1, 6, size=intervals)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "glucose_mg_dl": np.clip(glucose, 55, 280).round(1),
        "meal_carbs_g": meal_carbs.astype(int),
        "activity_minutes": activity,
        "sleep_hours": sleep.round(1),
        "stress_level": stress,
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated {len(df)} simulated readings.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_glucose_data()

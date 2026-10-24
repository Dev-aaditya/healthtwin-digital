
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "diabetes_model.joblib"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def predict_demo_outcome(values: dict) -> dict:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained model not found. Run train_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    # Keep feature names and order consistent with training.
    sample = pd.DataFrame(
        [{feature: values[feature] for feature in FEATURES}],
        columns=FEATURES,
    )

    score = float(model.predict_proba(sample)[0, 1])

    return {
        "model_score": round(score, 4),
        "model_class": int(model.predict(sample)[0]),
        "interpretation": (
            "Output from an educational classifier trained "
            "on a limited historical dataset. Not an "
            "individualized clinical risk estimate."
        ),
    }

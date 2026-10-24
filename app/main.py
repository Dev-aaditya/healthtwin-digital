
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.database import get_connection, init_db

from app.simulator import simulate_trajectory

from pathlib import Path
from fastapi.responses import FileResponse

from app.ml_model import predict_demo_outcome

import json
from pathlib import Path

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Happiest Health Digital Twin",
    description="A simulated health trajectory and prevention MVP",
    version="0.2.0",
    lifespan=lifespan,
)


def now():
    return datetime.now(timezone.utc).isoformat()


class ProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=18, le=120)
    height_cm: float = Field(gt=50, le=250)
    baseline_weight_kg: float = Field(gt=10, le=400)


class MeasurementCreate(BaseModel):
    weight_kg: Optional[float] = Field(default=None, gt=10, le=400)
    resting_hr_bpm: Optional[int] = Field(default=None, ge=20, le=250)
    glucose_mg_dl: Optional[float] = Field(default=None, gt=0, le=1000)




@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(
        Path(__file__).parent / "static" / "index.html"
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/profiles", status_code=201)
def create_profile(profile: ProfileCreate):
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO profiles
            (name, age, height_cm, baseline_weight_kg, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                profile.name,
                profile.age,
                profile.height_cm,
                profile.baseline_weight_kg,
                now(),
            ),
        )
        profile_id = cursor.lastrowid

        return dict(conn.execute(
            "SELECT * FROM profiles WHERE id = ?",
            (profile_id,),
        ).fetchone())


@app.get("/profiles/{profile_id}")
def get_profile(profile_id: int):
    with get_connection() as conn:
        profile = conn.execute(
            "SELECT * FROM profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()

        if profile is None:
            raise HTTPException(404, "Profile not found")

        return dict(profile)


@app.post("/profiles/{profile_id}/measurements", status_code=201)
def add_measurement(
    profile_id: int,
    measurement: MeasurementCreate,
):
    values = measurement.model_dump()

    if all(value is None for value in values.values()):
        raise HTTPException(
            400,
            "Provide at least one measurement.",
        )

    with get_connection() as conn:
        profile = conn.execute(
            "SELECT id FROM profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()

        if profile is None:
            raise HTTPException(404, "Profile not found")

        cursor = conn.execute(
            """
            INSERT INTO measurements
            (profile_id, recorded_at, weight_kg,
             resting_hr_bpm, glucose_mg_dl)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                now(),
                measurement.weight_kg,
                measurement.resting_hr_bpm,
                measurement.glucose_mg_dl,
            ),
        )

        record = conn.execute(
            "SELECT * FROM measurements WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

        return dict(record)


@app.get("/profiles/{profile_id}/measurements")
def get_measurements(profile_id: int):
    with get_connection() as conn:
        profile = conn.execute(
            "SELECT id FROM profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()

        if profile is None:
            raise HTTPException(404, "Profile not found")

        records = conn.execute(
            """
            SELECT * FROM measurements
            WHERE profile_id = ?
            ORDER BY recorded_at ASC, id ASC
            """,
            (profile_id,),
        ).fetchall()

        return {
            "profile_id": profile_id,
            "count": len(records),
            "measurements": [dict(row) for row in records],
        }



@app.post("/profiles/{profile_id}/simulate")
def run_simulation(
    profile_id: int,
    weeks: int = 12,
):
    if weeks < 1 or weeks > 52:
        raise HTTPException(
            400,
            "Weeks must be between 1 and 52.",
        )

    with get_connection() as conn:
        profile = conn.execute(
            "SELECT * FROM profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()

        if profile is None:
            raise HTTPException(404, "Profile not found")

        latest = conn.execute(
            """
            SELECT * FROM measurements
            WHERE profile_id = ?
            ORDER BY recorded_at DESC, id DESC
            LIMIT 1
            """,
            (profile_id,),
        ).fetchone()

        # Use the latest available measurement.
        # Fall back to the profile's baseline weight.
        weight = (
            latest["weight_kg"]
            if latest and latest["weight_kg"] is not None
            else profile["baseline_weight_kg"]
        )

        hr = (
            latest["resting_hr_bpm"]
            if latest
            else None
        )

        glucose = (
            latest["glucose_mg_dl"]
            if latest
            else None
        )

        result = simulate_trajectory(
            baseline_weight_kg=weight,
            baseline_hr_bpm=hr,
            baseline_glucose_mg_dl=glucose,
            weeks=weeks,
        )

        return {
            "profile_id": profile_id,
            "simulation_type": "illustrative",
            "disclaimer": (
                "Hypothetical scenario only. "
                "Not a clinical prediction."
            ),
            **result,
        }




class MLPredictionInput(BaseModel):
    pregnancies: int = Field(ge=0, le=30)
    glucose: float = Field(gt=0, le=500)
    blood_pressure: float = Field(gt=0, le=300)
    skin_thickness: float = Field(ge=0, le=150)
    insulin: float = Field(ge=0, le=3000)
    bmi: float = Field(gt=0, le=150)
    diabetes_pedigree_function: float = Field(ge=0, le=10)
    age: int = Field(ge=21, le=120)


@app.post("/ml/predict")
def ml_predict(data: MLPredictionInput):
    values = {
        "Pregnancies": data.pregnancies,
        "Glucose": data.glucose,
        "BloodPressure": data.blood_pressure,
        "SkinThickness": data.skin_thickness,
        "Insulin": data.insulin,
        "BMI": data.bmi,
        "DiabetesPedigreeFunction":
            data.diabetes_pedigree_function,
        "Age": data.age,
    }

    try:
        result = predict_demo_outcome(values)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    return {
        "model": "logistic_regression",
        "purpose": "educational demonstration",
        "dataset_population": (
            "Pima Indian women represented in the source dataset"
        ),
        **result,
        "medical_disclaimer": (
            "Not a diagnosis, screening recommendation, "
            "or substitute for professional medical advice."
        ),
    }



@app.get("/ml/metrics")
def get_ml_metrics():
    metrics_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "metrics.json"
    )

    if not metrics_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Metrics unavailable. Run train_model.py first.",
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    return {
        "model": "Logistic Regression",
        "evaluation": "Held-out test set",
        "metrics": metrics,
        "disclaimer": (
            "Preliminary evaluation on a limited dataset; "
            "not evidence of clinical validity or generalizability."
        ),
    }

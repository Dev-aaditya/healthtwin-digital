
from typing import Optional


def simulate_trajectory(
    baseline_weight_kg: float,
    baseline_hr_bpm: Optional[int],
    baseline_glucose_mg_dl: Optional[float],
    weeks: int = 12,
):
    """
    Generate illustrative trajectories for a demo digital twin.

    This is a rule-based visualization model, not a
    clinically validated prediction model.
    """

    baseline = []
    intervention = []

    for week in range(weeks + 1):
        baseline_point = {
            "week": week,
            "weight_kg": round(baseline_weight_kg, 2),
            "resting_hr_bpm": baseline_hr_bpm,
            "glucose_mg_dl": (
                round(baseline_glucose_mg_dl, 2)
                if baseline_glucose_mg_dl is not None
                else None
            ),
        }

        # Hypothetical demo scenario parameters:
        # These rates are invented for visualization only.
        intervention_point = {
            "week": week,
            "weight_kg": round(
                baseline_weight_kg - (0.1 * week), 2
            ),
            "resting_hr_bpm": (
                max(20, baseline_hr_bpm - (week // 3))
                if baseline_hr_bpm is not None
                else None
            ),
            "glucose_mg_dl": (
                round(
                    max(
                        1,
                        baseline_glucose_mg_dl - (0.5 * week),
                    ),
                    2,
                )
                if baseline_glucose_mg_dl is not None
                else None
            ),
        }

        baseline.append(baseline_point)
        intervention.append(intervention_point)

    return {
        "weeks": weeks,
        "baseline": baseline,
        "intervention": intervention,
        "assumptions": {
            "baseline": "All indicators remain constant.",
            "intervention": (
                "Illustrative changes: weight decreases by "
                "0.1 kg/week, resting heart rate decreases by "
                "1 BPM every 3 weeks, and glucose decreases "
                "by 0.5 mg/dL/week."
            ),
            "clinical_validity": (
                "Not clinically validated. Not a medical "
                "forecast or treatment recommendation."
            ),
        },
    }

# HealthTwin — Interactive Health Digital Twin Prototype


## Team details
- Team name: yadavaaditya26
- Team member(s): Aaditya Yadav, Agrim Tiwari
- Team size: 2

## College / incubator information
- College / institution: Lakshmi Narain College Of Technology Excellence
- Incubator / accelerator: Not applicable

## Project title
**HealthTwin: An Interactive Health Digital Twin Prototype**

## Problem statement
Health measurements can be recorded over time without an accessible way to explore trends or compare hypothetical future scenarios. HealthTwin is a prototype for organizing a participant's baseline profile and longitudinal measurements, visualizing an illustrative scenario alongside a stable baseline, and demonstrating a separate machine-learning classification pipeline.

## Healthcare use case
A demo participant has a baseline profile and timestamped measurements such as weight, resting heart rate, and glucose. Users can create/retrieve a profile, add/view measurements, compare a stable baseline with a hypothetical rule-based scenario, run a trained educational classifier, and inspect held-out test metrics.

**Responsible-use statement:** This is an educational prototype, not a clinical decision-support system. The simulator uses hand-coded illustrative rates, not a validated physiological model. The ML classifier is not clinically validated and must not be used to diagnose, screen, or make treatment decisions. Use synthetic/demo participant information only.

## Technical stack
- Python, JavaScript, HTML, CSS
- FastAPI, Uvicorn
- SQLite
- pandas, NumPy, scikit-learn, joblib
- Chart.js
- Pydantic / FastAPI request validation

## AI/ML model details
- Model: scikit-learn Logistic Regression pipeline with median imputation and feature scaling.
- Dataset: Pima Indians Diabetes dataset; eight input features and binary `Outcome` label.
- Split: Stratified 80/20 train/test split, `random_state=42`.
- Reported run: 154 held-out records; accuracy **73.4%**, balanced accuracy **72.7%**, ROC-AUC **0.813**.
- Confusion matrix: `[[75, 25], [16, 38]]`; rows are actual classes, columns predicted classes.
- Limitations: one held-out split on a small, population-specific historical dataset. Not external validation, clinical validation, or evidence of generalizability. The score is not a calibrated clinical risk estimate.

## Architecture
Browser dashboard (HTML/CSS/JavaScript + Chart.js) → FastAPI REST API → SQLite profile/measurement storage; rule-based scenario simulator; scikit-learn classifier and saved metrics → dashboard charts/results. See `architecture.pdf`.

## API endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/health` | Health check |
| POST | `/profiles` | Create profile |
| GET | `/profiles/{profile_id}` | Retrieve profile |
| POST | `/profiles/{profile_id}/measurements` | Add measurement |
| GET | `/profiles/{profile_id}/measurements` | Measurement history |
| POST | `/profiles/{profile_id}/simulate?weeks=12` | Illustrative trajectories |
| POST | `/ml/predict` | Educational classifier inference |
| GET | `/ml/metrics` | Saved evaluation metrics |

## Local setup
Requirements: Python 3.10+.

1. Clone the public repository:
   ```bash
   git clone https://github.com/Dev-aaditya/healthtwin-digital
   cd 'Your Local Repository address'
   ```
2. Create/activate a virtual environment (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
4. Ensure `data/diabetes.csv` is present with expected columns. Check dataset terms before redistribution. If it cannot be redistributed, document how to obtain it instead.
5. Train the model:
   ```powershell
   python train_model.py
   ```
6. Start the app:
   ```powershell
   python -m uvicorn app.main:app --reload
   ```
7. Open `http://127.0.0.1:8000/` and `http://127.0.0.1:8000/docs`.

## Demo video (mandatory)
**Unlisted YouTube video, 15–20 minutes:** https://youtu.be/oQ3lTJRPaYg  
Add the final URL here and to the submission form. Verify reviewers can access it.

## Open-source license
The project code is intended to use the MIT License; see `LICENSE`. Replace its copyright placeholder. The code license does not automatically cover third-party datasets or other assets; verify those terms separately.

## Included submission materials
- `architecture.pdf` — architecture diagram.
- `presentation.pptx` — project presentation.
- `LICENSE` — MIT license template.
- `demo_script.md` — suggested narration and demo sequence.
- `submission_checklist.md` — final checklist.

## Limitations and future work
Scenario rates are illustrative, not validated intervention effects. The ML dataset is small and population-specific; no external validation has been performed. Do not enter real patient data or use the app for medical decisions. Future work includes appropriate longitudinal data, external validation, calibration, fairness analysis, privacy/security controls, and clinical review.

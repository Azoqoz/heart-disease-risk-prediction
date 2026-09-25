# Heart Disease Risk Prediction

An educational heart-disease screening application with a Next.js workstation interface and a FastAPI inference service. The API loads one calibrated inference pipeline containing preprocessing, XGBoost, and sigmoid probability calibration; model artifacts never enter the browser.

> This project is an educational demonstration. Its output is not a medical diagnosis and does not replace evaluation by a qualified healthcare professional.

## Architecture

```text
Next.js clinical workstation
        │ semantic patient inputs
        ▼
FastAPI validation and encoding
        │ fixed 13-feature vector
        ▼
StandardScaler → XGBoost classifier → sigmoid calibration
        │ calibrated class-1 model estimate
        ▼
50% interpretation threshold → API response
```

The backend is the sole owner of categorical encoding, feature ordering, scaling, and inference. The frontend sends human-readable choices and renders the returned model estimate.

## Project structure

```text
frontend/                 Next.js App Router application
backend/app/              FastAPI service and inference contract
backend/tests/            API regression tests
tests/                    Artifact and inference regression tests
app.py                    Retired Streamlit implementation, kept for history
heart_model_calibrated.joblib
                          Production calibrated inference pipeline
heart_model.joblib        Legacy pre-calibration model, retained for comparison
scaler.joblib             Legacy scaler, retained for comparison
train_model.py            Reproducible deduplication, training, and calibration workflow
```

## Local development

### Backend

Python 3.11 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

The API is available at `http://localhost:8000` and exposes:

- `GET /health` — service and artifact readiness
- `GET /metadata` — safe UI metadata only
- `POST /predict` — validated model inference

### Frontend

Node.js 20 or newer is recommended.

```powershell
cd frontend
npm install
npm run dev
```

The web application is available at `http://localhost:3000`. It uses `http://localhost:8000` by default. To use another API origin, copy `frontend/.env.example` to `frontend/.env.local` and set:

```text
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

For deployed environments, configure the backend `CORS_ORIGINS` variable as a comma-separated list of allowed frontend origins.

## Verification

Run the artifact/inference regression suite:

```powershell
.venv\Scripts\python -B -m unittest discover -s tests -v
```

Run the FastAPI suite:

```powershell
.venv\Scripts\python -B -m pytest backend\tests -v -p no:cacheprovider
```

Verify the frontend:

```powershell
cd frontend
npm run lint
npm run build
```

The regression tests lock the saved-artifact contract, exact feature order, class-1 probability direction, and 50% interpretation threshold. They are not claims of clinical performance.

## Inference contract

The model input order is fixed:

```text
age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal
```

The API passes the ordered dataframe to the saved inference pipeline and returns `predict_proba(input_data)[0][1]`. Results at or above `0.50` are labeled `Higher probability`; results below `0.50` are labeled `Lower probability`.

## Model training

`train_model.py` removes only exact duplicate rows, creates a stratified 80/20 split with `random_state=42`, and keeps the test partition outside model fitting and calibration. A five-fold stratified `CalibratedClassifierCV` fits sigmoid calibration around a conservative StandardScaler/XGBoost pipeline and saves the complete inference path as one artifact.

The original Streamlit application, analysis notebook, model, and scaler remain in the repository as project history. Normal application startup does not retrain the model.

## Limitations

- The model has not been clinically validated.
- Its calibrated output remains a model-estimated probability, not a clinically validated disease probability.
- Dataset bias and limited representation may affect results across patient groups.
- The application is not a medical device and must not be used for diagnosis, treatment, or healthcare decisions.
- Production healthcare use would require independent validation, security, privacy, governance, and regulatory review.

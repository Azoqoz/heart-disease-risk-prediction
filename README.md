# Heart Disease Risk Prediction

An educational machine-learning screening application with a modern Next.js clinical workstation, a FastAPI inference service, and a calibrated XGBoost prediction pipeline.

The application collects 13 clinical inputs, validates and encodes them on the backend, runs them through a saved preprocessing and model-calibration pipeline, and returns a model-estimated probability through a clear screening interface.

> This project is an educational demonstration. Its output is not a medical diagnosis and does not replace evaluation by a qualified healthcare professional.

![Python](https://img.shields.io/badge/Language-Python-blue)
![Next.js](https://img.shields.io/badge/Frontend-Next.js-black)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E)
![Pytest](https://img.shields.io/badge/Testing-Pytest-yellow)

---

## Live Demo

- **Web Application:** https://heart-disease-risk-prediction-eta.vercel.app
- **API Documentation:** https://heart-disease-risk-prediction-qjn4.onrender.com/docs
- **API Health:** https://heart-disease-risk-prediction-qjn4.onrender.com/health

---

## Overview

Heart Disease Risk Prediction is an educational machine-learning project designed to demonstrate a complete inference workflow rather than only model training.

The current version separates the application into:

- A Next.js frontend
- A FastAPI backend
- A calibrated XGBoost inference pipeline
- Backend-controlled feature encoding and validation
- Automated regression and API tests
- Separate frontend and backend deployments

The browser never loads the machine-learning artifact directly.

Instead, the frontend collects human-readable patient information and sends it to the FastAPI service.

The backend is responsible for:

- Input validation
- Categorical encoding
- Feature ordering
- Preprocessing
- Model inference
- Probability calibration
- Threshold interpretation

---

## Key Features

- Modern multi-step clinical screening interface
- 13-feature heart-disease input workflow
- Next.js frontend
- FastAPI inference API
- XGBoost classification
- StandardScaler preprocessing
- Sigmoid probability calibration
- Fixed backend-controlled feature order
- Human-readable frontend categories
- Backend-only model execution
- Input review before inference
- Optional clinical-signal defaults
- Higher / Lower probability interpretation
- Visual 50% decision threshold
- API readiness indicator
- Health and metadata endpoints
- Swagger API documentation
- Regression tests for saved model behavior
- FastAPI tests using Pytest
- Separate frontend and backend deployment
- Educational medical-use disclaimer

---

## Screening Workflow

1. Enter basic patient information.
2. Enter recorded cardiovascular measurements.
3. Select symptom indicators.
4. Adjust optional clinical signals or use predefined defaults.
5. Review all 13 inputs.
6. Send the validated profile to the FastAPI backend.
7. Convert semantic inputs into the fixed model feature vector.
8. Run preprocessing and calibrated XGBoost inference.
9. Return the model-estimated probability.
10. Display the result relative to the configured 50% threshold.

Example workflow:

```text
Patient Profile
      |
      v
Vital Measurements
      |
      v
Symptoms
      |
      v
Optional Clinical Signals
      |
      v
Review Inputs
      |
      v
FastAPI Validation
      |
      v
Feature Encoding
      |
      v
StandardScaler
      |
      v
XGBoost
      |
      v
Sigmoid Calibration
      |
      v
Model-Estimated Probability
      |
      v
Higher / Lower Probability
```

---

## System Architecture

```text
┌─────────────────────────────┐
│       Next.js Frontend      │
│                             │
│  Patient                    │
│  Vitals                     │
│  Symptoms                   │
│  Clinical Signals           │
└──────────────┬──────────────┘
               │
               │ semantic JSON payload
               ▼
┌─────────────────────────────┐
│        FastAPI API          │
│                             │
│  Request validation         │
│  Semantic encoding          │
│  Feature ordering           │
│  Input safety checks        │
└──────────────┬──────────────┘
               │
               │ 13-feature dataframe
               ▼
┌─────────────────────────────┐
│   Calibrated ML Pipeline    │
│                             │
│  StandardScaler             │
│       ↓                     │
│  XGBoost Classifier         │
│       ↓                     │
│  Sigmoid Calibration        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Prediction Response     │
│                             │
│  Probability                │
│  Interpretation             │
│  Threshold                  │
└─────────────────────────────┘
```

The backend is the sole owner of categorical encoding, feature ordering, preprocessing, and inference.

The frontend does not construct the final model feature vector directly.

---

## Model Input

The model uses 13 features in the following fixed order:

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

The backend explicitly constructs this feature order before passing data into the inference pipeline.

This prevents frontend object ordering or semantic labels from changing the model contract.

---

## Input Groups

### Patient

- Age
- Sex

### Vitals

- Resting blood pressure
- Cholesterol
- Maximum heart rate

### Symptoms

- Chest pain type
- Exercise-induced angina
- High fasting blood sugar

### Clinical Signals

- Resting ECG
- ST depression
- ST slope
- Number of major vessels
- Thalassemia

The clinical-signal section can also use predefined default values when the user does not provide additional measurements.

---

## Inference Pipeline

The production artifact contains the complete fitted inference path.

```text
Input
  |
  v
StandardScaler
  |
  v
XGBoost Classifier
  |
  v
Sigmoid Probability Calibration
  |
  v
Calibrated Class-1 Probability
```

The production FastAPI backend loads:

```text
heart_model_calibrated.joblib
```

The frontend never receives or loads the model artifact.

---

## Probability Interpretation

The API returns the model-estimated probability for class `1`.

```python
predict_proba(input_data)[0][1]
```

The current interpretation threshold is:

```text
50%
```

Results are displayed as:

```text
Probability >= 50%
→ Higher probability
```

```text
Probability < 50%
→ Lower probability
```

This threshold is an application-level interpretation rule.

It is not a clinically selected diagnostic threshold.

---

## Model Training

The current training workflow is implemented in:

```text
train_model.py
```

The workflow:

1. Loads the heart-disease dataset.
2. Removes exact duplicate rows.
3. Separates features and target.
4. Creates a stratified 80/20 train/test split.
5. Keeps the test partition outside model fitting.
6. Fits preprocessing only on training data.
7. Trains a conservative XGBoost classifier.
8. Applies five-fold stratified sigmoid calibration.
9. Evaluates the final artifact on the untouched held-out set.
10. Saves the complete production inference pipeline as one artifact.

The split uses:

```text
random_state = 42
```

---

## Duplicate Leakage Correction

The original dataset contained extensive exact duplication.

Before the corrected pipeline:

```text
Original rows: 1,025
Exact duplicates removed: 723
Unique rows: 302
```

The original random train/test split therefore contained severe duplicate leakage.

The corrected workflow removes exact duplicates before splitting.

The final deduplicated split contains:

```text
Training rows: 241
Test rows: 61

Training class distribution:
0: 110
1: 131

Test class distribution:
0: 28
1: 33

Exact train/test row overlap: 0
Exact feature-vector overlap: 0
```

Old performance results produced from the leaked split are not treated as valid model-performance evidence.

---

## Held-Out Evaluation

The current model was evaluated on the untouched 61-row deduplicated test set.

| Metric | Result |
|---|---:|
| Accuracy | 0.7705 |
| Precision | 0.7879 |
| Recall | 0.7879 |
| F1 Score | 0.7879 |
| ROC-AUC | 0.8669 |
| Log Loss | 0.4618 |
| Brier Score | 0.1504 |

Confusion matrix:

```text
[[21, 7],
 [ 7, 26]]
```

These results describe performance only on the current held-out dataset.

They are not evidence of clinical validity.

---

## Probability Calibration

The project uses:

```text
CalibratedClassifierCV
method = sigmoid
```

Calibration was added to reduce excessively extreme probability outputs.

Observed probability range on the held-out test set:

```text
Before calibration:
1.53% – 97.94%

After calibration:
6.27% – 92.50%
```

Extreme predictions at or below 5% or at or above 95% changed from:

```text
11 / 61
```

to:

```text
0 / 61
```

Calibration reduced extreme confidence.

However, held-out Log Loss and Brier Score did not improve on this small test set, so the project does not claim that calibration improved aggregate probabilistic accuracy.

---

## API

The FastAPI service exposes three main endpoints.

### Health

```http
GET /health
```

Returns service and model readiness.

Example:

```json
{
  "status": "ready",
  "model_ready": true
}
```

### Metadata

```http
GET /metadata
```

Returns safe application metadata required by the frontend.

### Predict

```http
POST /predict
```

Receives the validated patient profile and returns the model-estimated probability and interpretation.

Interactive API documentation is available at:

https://heart-disease-risk-prediction-qjn4.onrender.com/docs

---

## Frontend

The frontend is built with Next.js and uses a guided four-step workflow.

```text
01 Patient
02 Vitals
03 Symptoms
04 Signals
```

The application also includes:

- Input review
- Model result visualization
- Probability threshold indicator
- API readiness status
- Higher / Lower probability states
- Educational disclaimer

The frontend sends semantic values such as:

```text
Female
Male
Yes
No
Mild or unusual chest pain
Chest pain during physical activity
```

The backend converts those values into the numeric representation expected by the model.

---

## Tech Stack

| Category | Technology |
|---|---|
| Frontend | Next.js |
| Frontend language | TypeScript |
| Styling | CSS |
| Backend | FastAPI |
| Backend language | Python |
| Model | XGBoost |
| Preprocessing | StandardScaler |
| Calibration | CalibratedClassifierCV |
| Validation | Pydantic |
| ML utilities | Scikit-learn |
| Model serialization | Joblib |
| Backend testing | Pytest |
| Regression testing | Python unittest |
| Frontend deployment | Vercel |
| Backend deployment | Render |

---

## Project Structure

```text
heart-disease-risk-prediction/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── inference.py
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── tests/
│   │   └── test_api.py
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── globals.css
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx
│   │   │
│   │   ├── components/
│   │   │   └── ClinicalWorkbench.tsx
│   │   │
│   │   └── lib/
│   │       ├── api.ts
│   │       └── types.ts
│   │
│   ├── .env.example
│   ├── package.json
│   └── tsconfig.json
│
├── tests/
│   └── test_inference_regression.py
│
├── heart_model_calibrated.joblib
├── heart_model.joblib
├── scaler.joblib
├── train_model.py
├── app.py
├── README.md
└── .gitignore
```

---

## Local Development

### Backend

Python 3.11 is recommended.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r backend\requirements.txt
```

Start the API:

```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## Frontend Development

Node.js 20 or newer is recommended.

```powershell
cd frontend
npm install
npm run dev
```

The application will be available at:

```text
http://localhost:3000
```

By default, the frontend connects to:

```text
http://localhost:8000
```

To configure another backend, copy:

```text
frontend/.env.example
```

to:

```text
frontend/.env.local
```

and set:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

## Environment Configuration

### Frontend

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### Backend Deployment

```env
CORS_ORIGINS=https://example.vercel.app,http://localhost:3000
```

Multiple allowed origins are comma-separated.

---

## Testing

### Artifact and Inference Regression Tests

```powershell
.venv\Scripts\python -B -m unittest discover -s tests -v
```

These tests verify:

- Saved model artifact availability
- Expected feature order
- Model class direction
- Probability behavior
- Threshold interpretation
- Regression consistency

### FastAPI Tests

```powershell
.venv\Scripts\python -B -m pytest backend\tests -v -p no:cacheprovider
```

These tests cover:

- API health
- Model readiness
- Request validation
- Prediction responses
- Invalid inputs
- Probability output

### Frontend Verification

```powershell
cd frontend
npm run lint
npm run build
```

---

## Deployment

### Frontend — Vercel

Configuration:

```text
Framework: Next.js
Root directory: frontend
```

Environment variable:

```env
NEXT_PUBLIC_API_BASE_URL=https://heart-disease-risk-prediction-qjn4.onrender.com
```

Production frontend:

https://heart-disease-risk-prediction-eta.vercel.app

### Backend — Render

Runtime:

```text
Python 3
```

Build command:

```bash
python -m pip install -r backend/requirements.txt
```

Start command:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```env
PYTHON_VERSION=3.11.11
CORS_ORIGINS=https://heart-disease-risk-prediction-eta.vercel.app,http://localhost:3000,http://127.0.0.1:3000
```

Production API:

https://heart-disease-risk-prediction-qjn4.onrender.com

---

## Legacy Implementation

The repository still contains the original Streamlit implementation:

```text
app.py
```

It is retained for project history and comparison.

The current production application uses:

```text
Next.js + FastAPI
```

The legacy model and scaler are also retained:

```text
heart_model.joblib
scaler.joblib
```

The production backend instead loads:

```text
heart_model_calibrated.joblib
```

Normal application startup does not retrain the model.

---

## Current Limitations

- The model has not been clinically validated.
- The output is a model-estimated probability, not a clinically validated disease probability.
- Only 302 unique observations remain after exact deduplication.
- The held-out test set contains only 61 observations.
- No external validation dataset is currently used.
- Dataset bias and limited representation may affect model behavior.
- Some transformed categorical codes have limited provenance documentation.
- The 50% interpretation threshold has not been clinically selected.
- The application is not a medical device.
- It must not be used for diagnosis, treatment, or healthcare decisions.
- Production healthcare use would require independent validation, privacy controls, security controls, governance, and regulatory review.

---

## Future Improvements

- Validate the model on an external dataset
- Expand the training dataset
- Improve categorical feature documentation
- Evaluate additional calibration strategies
- Evaluate threshold selection under defined objectives
- Add model explainability
- Add feature-level contribution visualization
- Add automated CI testing
- Add model versioning
- Add artifact metadata
- Add inference monitoring
- Add Docker support
- Add production authentication
- Add rate limiting
- Add structured observability
- Add retraining and model-release workflows

---

## Why This Project Matters

This project demonstrates more than training a classifier.

It shows an end-to-end machine-learning application that includes:

- Data-quality auditing
- Duplicate leakage detection
- Leakage-aware dataset splitting
- Feature preprocessing
- XGBoost classification
- Probability calibration
- Held-out evaluation
- Model artifact management
- Backend inference architecture
- API validation
- Frontend / backend separation
- Model deployment
- CORS configuration
- Regression testing
- API testing
- Modern Next.js interface development
- Deployment with Vercel and Render
- Clear model limitations and responsible communication

The project demonstrates how a machine-learning model can be integrated into a complete software system while keeping preprocessing, feature semantics, inference, and model artifacts controlled by the backend.

---

## Author

Developed by [Azoqoz](https://github.com/Azoqoz).

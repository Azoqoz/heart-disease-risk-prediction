# Heart Disease Risk Prediction

An educational machine-learning screening application with a modern Next.js clinical workstation, a FastAPI inference service, and a calibrated XGBoost prediction pipeline.

The application collects 13 clinical inputs, validates and encodes them on the backend, runs them through a saved preprocessing and model-calibration pipeline, and returns a model-estimated probability through a clear screening interface.

> This project is an educational demonstration. Its output is not a medical diagnosis and does not replace evaluation by a qualified healthcare professional.

![Python](https://img.shields.io/badge/Language-Python-blue)
![Next.js](https://img.shields.io/badge/Frontend-Next.js-black)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E)
![Testing](https://img.shields.io/badge/Testing-Pytest-yellow)

---

## Live Demo

- **Web Application:** https://heart-disease-risk-prediction-eta.vercel.app
- **API Documentation:** https://heart-disease-risk-prediction-qjn4.onrender.com/docs
- **API Health:** https://heart-disease-risk-prediction-qjn4.onrender.com/health

---

## Overview

Heart Disease Risk Prediction is an educational machine-learning project designed to demonstrate an end-to-end inference workflow rather than only model training.

The current version separates the application into:

- A Next.js frontend
- A FastAPI backend
- A calibrated XGBoost inference pipeline
- Backend-controlled feature encoding and validation
- Automated regression and API tests
- Hosted frontend and backend deployments

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
- Review screen before inference
- Optional clinical-signal defaults
- Higher / Lower probability interpretation
- Visual 50% decision threshold
- Model health endpoint
- Metadata endpoint
- Swagger API documentation
- Regression tests for saved model behavior
- API tests using Pytest
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

Example flow:

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

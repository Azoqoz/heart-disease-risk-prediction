import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.inference import (
    FEATURE_ORDER,
    THRESHOLD,
    build_input_frame,
    get_model_service,
    interpret_probability,
)
from backend.app.main import app
from backend.app.schemas import PredictionRequest


client = TestClient(app)

PRODUCTION_DEFAULTS = {
    "age": 25,
    "sex": "Male",
    "cp": "Chest pain during physical activity",
    "trestbps": 120,
    "chol": 200,
    "fbs": "No",
    "restecg": 0,
    "thalach": 150,
    "exang": "No",
    "oldpeak": 1.0,
    "slope": 0,
    "ca": 0,
    "thal": 0,
}

VALIDATED_REFERENCE = {
    "age": 52,
    "sex": "Male",
    "cp": "Chest pain during physical activity",
    "trestbps": 125,
    "chol": 212,
    "fbs": "No",
    "restecg": 1,
    "thalach": 168,
    "exang": "No",
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3,
}

CORRECTED_UI_PROFILE = {
    "age": 35,
    "sex": "Female",
    "cp": "Mild or unusual chest pain",
    "trestbps": 110,
    "chol": 180,
    "fbs": "No",
    "restecg": 1,
    "thalach": 180,
    "exang": "No",
    "oldpeak": 0.0,
    "slope": 2,
    "ca": 0,
    "thal": 2,
}


def assert_probability(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)


def test_health_endpoint_reports_model_readiness():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_ready": True}


def test_metadata_endpoint_exposes_only_safe_model_context():
    response = client.get("/metadata")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "XGBoost"
    assert payload["feature_count"] == 13
    assert payload["threshold"] == 0.50
    assert payload["educational_only"] is True
    assert "Accuracy" not in response.text
    assert "Precision" not in response.text
    assert "Recall" not in response.text


def test_production_defaults_regression_probability():
    response = client.post("/predict", json=PRODUCTION_DEFAULTS)
    assert response.status_code == 200
    result = response.json()
    assert_probability(result["probability"], 0.6987311075316515)
    assert result["interpretation"] == "Higher probability"


def test_validated_reference_regression_probability():
    response = client.post("/predict", json=VALIDATED_REFERENCE)
    assert response.status_code == 200
    result = response.json()
    assert_probability(result["probability"], 0.10851857218505596)
    assert result["interpretation"] == "Lower probability"


def test_corrected_ui_profile_regression_probability():
    response = client.post("/predict", json=CORRECTED_UI_PROFILE)
    assert response.status_code == 200
    result = response.json()
    assert_probability(result["probability"], 0.9239147891015914)
    assert result["interpretation"] == "Higher probability"


def test_invalid_input_is_rejected():
    invalid = {**PRODUCTION_DEFAULTS, "age": 101}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_extra_input_is_rejected():
    invalid = {**PRODUCTION_DEFAULTS, "unexpected": 1}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_non_finite_input_is_rejected_before_inference():
    with pytest.raises(ValidationError):
        PredictionRequest(**{**PRODUCTION_DEFAULTS, "oldpeak": float("nan")})


def test_exact_13_feature_contract_and_order():
    assert len(PredictionRequest.model_fields) == 13
    frame = build_input_frame(PredictionRequest(**PRODUCTION_DEFAULTS))
    assert tuple(frame.columns) == FEATURE_ORDER


def test_service_uses_class_one_probability():
    payload = PredictionRequest(**VALIDATED_REFERENCE)
    service = get_model_service()
    frame = build_input_frame(payload)
    expected = float(service.model.predict_proba(frame)[0][1])
    actual = service.predict(payload).probability
    assert_probability(actual, expected)
    assert math.isfinite(actual)
    assert 0.0 <= actual <= 1.0


def test_threshold_interpretation_contract():
    assert THRESHOLD == 0.50
    assert interpret_probability(0.499999) == "Lower probability"
    assert interpret_probability(0.50) == "Higher probability"
    assert interpret_probability(0.9) == "Higher probability"

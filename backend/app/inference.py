from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .schemas import PredictionRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "heart_model_calibrated.joblib"

FEATURE_ORDER = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)

THRESHOLD = 0.50
MODEL_NAME = "XGBoost"
DISCLAIMER = (
    "Educational screening tool only. This model output is not a medical "
    "diagnosis and does not replace evaluation by a qualified healthcare "
    "professional."
)

SEX_MAPPING = {"Male": 1, "Female": 0}
CHEST_PAIN_MAPPING = {
    "Chest pain during physical activity": 0,
    "Mild or unusual chest pain": 1,
    "Chest pain not related to the heart": 2,
    "No chest pain symptoms": 3,
}
BINARY_MAPPING = {"No": 0, "Yes": 1}


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    percentage: float
    interpretation: str


def interpret_probability(probability: float) -> str:
    return "Higher probability" if probability >= THRESHOLD else "Lower probability"


def build_input_frame(payload: PredictionRequest) -> pd.DataFrame:
    values = [
        payload.age,
        SEX_MAPPING[payload.sex],
        CHEST_PAIN_MAPPING[payload.cp],
        payload.trestbps,
        payload.chol,
        BINARY_MAPPING[payload.fbs],
        payload.restecg,
        payload.thalach,
        BINARY_MAPPING[payload.exang],
        payload.oldpeak,
        payload.slope,
        payload.ca,
        payload.thal,
    ]
    return pd.DataFrame([values], columns=FEATURE_ORDER)


class ModelService:
    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self._validate_artifacts()

    def _validate_artifacts(self) -> None:
        if self.model.classes_.tolist() != [0, 1]:
            raise RuntimeError("Model classes do not match the validated [0, 1] contract")
        if self.model.n_features_in_ != len(FEATURE_ORDER):
            raise RuntimeError("Model feature count does not match the validated contract")
        if tuple(self.model.feature_names_in_.tolist()) != FEATURE_ORDER:
            raise RuntimeError("Model feature order does not match the validated contract")

    def predict(self, payload: PredictionRequest) -> PredictionResult:
        input_data = build_input_frame(payload)
        if not np.isfinite(input_data.to_numpy(dtype=float)).all():
            raise ValueError("Model input contains a non-finite value")

        probability = float(self.model.predict_proba(input_data)[0][1])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("Model returned an invalid class-1 probability")

        return PredictionResult(
            probability=probability,
            percentage=probability * 100,
            interpretation=interpret_probability(probability),
        )


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService()

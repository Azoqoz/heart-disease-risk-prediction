import math
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "heart_model_calibrated.joblib"
INFERENCE_PATH = PROJECT_ROOT / "backend" / "app" / "inference.py"

FEATURE_ORDER = [
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
]

REGRESSION_CASES = {
    "production_defaults": {
        "values": [25, 1, 0, 120, 200, 0, 0, 150, 0, 1.0, 0, 0, 0],
        "class_1_probability": 0.6987311075316515,
    },
    "validated_reference": {
        "values": [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3],
        "class_1_probability": 0.10851857218505596,
    },
    "corrected_ui_profile": {
        "values": [35, 0, 1, 110, 180, 0, 1, 180, 0, 0.0, 2, 0, 2],
        "class_1_probability": 0.9239147891015914,
    },
}


class InferenceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(MODEL_PATH)
        cls.inference_source = INFERENCE_PATH.read_text(encoding="utf-8")

    def test_saved_calibrated_pipeline_loads_successfully(self):
        self.assertIsInstance(self.model, CalibratedClassifierCV)
        self.assertEqual(self.model.classes_.tolist(), [0, 1])
        self.assertEqual(self.model.method, "sigmoid")

    def test_pipeline_contains_scaling_and_xgboost(self):
        self.assertIsInstance(self.model.estimator, Pipeline)
        self.assertIsInstance(
            self.model.estimator.named_steps["scaler"],
            StandardScaler,
        )
        self.assertIsInstance(
            self.model.estimator.named_steps["classifier"],
            XGBClassifier,
        )

    def test_artifact_expects_exact_feature_contract(self):
        self.assertEqual(self.model.n_features_in_, 13)
        self.assertEqual(self.model.feature_names_in_.tolist(), FEATURE_ORDER)

    def test_backend_uses_class_one_probability_without_external_scaling(self):
        self.assertIn(
            "self.model.predict_proba(input_data)[0][1]",
            self.inference_source,
        )
        self.assertNotIn("SCALER_PATH", self.inference_source)

    def test_fixed_class_one_probabilities(self):
        for case_name, case in REGRESSION_CASES.items():
            with self.subTest(case=case_name):
                input_data = pd.DataFrame(
                    [case["values"]],
                    columns=FEATURE_ORDER,
                )
                probability = float(self.model.predict_proba(input_data)[0][1])

                self.assertTrue(
                    math.isclose(
                        probability,
                        case["class_1_probability"],
                        rel_tol=1e-6,
                        abs_tol=1e-8,
                    ),
                    msg=(
                        f"{case_name} class-1 probability changed: "
                        f"expected {case['class_1_probability']!r}, "
                        f"got {probability!r}"
                    ),
                )

    def test_threshold_and_interpretation_remain_unchanged(self):
        self.assertIn("THRESHOLD = 0.50", self.inference_source)
        self.assertIn(
            '"Higher probability" if probability >= THRESHOLD else "Lower probability"',
            self.inference_source,
        )

    def test_model_performance_claims_are_not_reintroduced(self):
        repository_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                PROJECT_ROOT / "README.md",
                PROJECT_ROOT / "app.py",
                PROJECT_ROOT / "frontend" / "src" / "components" / "ClinicalWorkbench.tsx",
            )
        )
        for claim in (
            "Accuracy: 98.54%",
            "Precision: 100%",
            "Recall: 97.09%",
            "F1 Score: 98.52%",
        ):
            with self.subTest(claim=claim):
                self.assertNotIn(claim, repository_text)


if __name__ == "__main__":
    unittest.main()

import math
import unittest
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"
MODEL_PATH = PROJECT_ROOT / "heart_model.joblib"
SCALER_PATH = PROJECT_ROOT / "scaler.joblib"

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
        "class_1_probability": 0.9263016581535339,
    },
    "validated_reference": {
        "values": [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3],
        "class_1_probability": 0.0048185959458351135,
    },
}


class InferenceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(MODEL_PATH)
        cls.scaler = joblib.load(SCALER_PATH)
        cls.app_source = APP_PATH.read_text(encoding="utf-8")

    def test_saved_artifacts_load_successfully(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.scaler)

    def test_model_is_binary_xgboost_with_expected_classes(self):
        self.assertEqual(type(self.model).__name__, "XGBClassifier")
        self.assertEqual(self.model.classes_.tolist(), [0, 1])

    def test_artifacts_expect_exactly_13_features(self):
        self.assertEqual(self.model.n_features_in_, 13)
        self.assertEqual(self.scaler.n_features_in_, 13)

    def test_scaler_feature_order_matches_production_contract(self):
        self.assertEqual(self.scaler.feature_names_in_.tolist(), FEATURE_ORDER)

    def test_app_uses_class_one_probability(self):
        self.assertIn(
            "model.predict_proba(input_scaled)[0][1]",
            self.app_source,
        )
        self.assertNotIn(
            "model.predict_proba(input_scaled)[0][0]",
            self.app_source,
        )

    def test_fixed_class_one_probabilities(self):
        for case_name, case in REGRESSION_CASES.items():
            with self.subTest(case=case_name):
                input_data = pd.DataFrame(
                    [case["values"]],
                    columns=FEATURE_ORDER,
                )
                input_scaled = self.scaler.transform(input_data)
                probability = float(
                    self.model.predict_proba(input_scaled)[0][1]
                )

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

    def test_interpretation_threshold_and_labels(self):
        self.assertIn(
            "is_higher_probability = risk_probability >= 0.50",
            self.app_source,
        )
        self.assertIn(
            'classification = "Higher probability" if '
            'is_higher_probability else "Lower probability"',
            self.app_source,
        )

    def test_app_does_not_restore_misleading_performance_claims(self):
        for claim in (
            "Accuracy: 98.54%",
            "Precision: 100%",
            "Recall: 97.09%",
            "F1 Score: 98.52%",
        ):
            with self.subTest(claim=claim):
                self.assertNotIn(claim, self.app_source)


if __name__ == "__main__":
    unittest.main()

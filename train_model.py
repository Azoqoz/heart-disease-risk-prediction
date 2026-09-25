from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "heart.csv"
ARTIFACT_PATH = PROJECT_ROOT / "heart_model_calibrated.joblib"
LEGACY_MODEL_PATH = PROJECT_ROOT / "heart_model.joblib"
LEGACY_SCALER_PATH = PROJECT_ROOT / "scaler.joblib"

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

REFERENCE_PROFILE = [35, 0, 1, 110, 180, 0, 1, 180, 0, 0.0, 2, 0, 2]
RANDOM_STATE = 42
TEST_SIZE = 0.20
CALIBRATION_FOLDS = 5


def build_base_pipeline() -> Pipeline:
    classifier = XGBClassifier(
        n_estimators=120,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def build_calibrated_model() -> CalibratedClassifierCV:
    calibration_cv = StratifiedKFold(
        n_splits=CALIBRATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return CalibratedClassifierCV(
        estimator=build_base_pipeline(),
        method="sigmoid",
        cv=calibration_cv,
    )


def class_counts(values: pd.Series) -> dict[int, int]:
    return {
        int(label): int(count)
        for label, count in values.value_counts().sort_index().items()
    }


def print_reference_result(label: str, model, input_data: pd.DataFrame) -> None:
    probability = float(model.predict_proba(input_data)[0][1])
    prediction = int(model.predict(input_data)[0])
    print(
        f"{label}: class={prediction}, "
        f"class_1_probability={probability:.12f}"
    )


def main() -> None:
    original = pd.read_csv(DATASET_PATH)
    expected_columns = FEATURE_ORDER + ["target"]
    if original.columns.tolist() != expected_columns:
        raise ValueError(
            "Dataset columns do not match the validated 13-feature contract"
        )

    duplicate_count = int(original.duplicated().sum())
    deduplicated = original.drop_duplicates().reset_index(drop=True)

    print("Dataset")
    print(f"  original_rows: {len(original)}")
    print(f"  exact_duplicates: {duplicate_count}")
    print(f"  unique_rows: {len(deduplicated)}")
    print(f"  class_counts_before: {class_counts(original['target'])}")
    print(f"  class_counts_after: {class_counts(deduplicated['target'])}")

    X = deduplicated[FEATURE_ORDER]
    y = deduplicated["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_rows = set(
        map(tuple, pd.concat([X_train, y_train], axis=1).to_numpy().tolist())
    )
    test_rows = set(
        map(tuple, pd.concat([X_test, y_test], axis=1).to_numpy().tolist())
    )
    overlap_count = len(train_rows.intersection(test_rows))
    train_feature_rows = set(map(tuple, X_train.to_numpy().tolist()))
    test_feature_rows = set(map(tuple, X_test.to_numpy().tolist()))
    feature_overlap_count = len(train_feature_rows.intersection(test_feature_rows))
    if overlap_count:
        raise RuntimeError("An exact observation appears in both train and test")

    print("\nSplit")
    print(f"  training_size: {len(X_train)}")
    print(f"  test_size: {len(X_test)}")
    print(f"  training_class_counts: {class_counts(y_train)}")
    print(f"  test_class_counts: {class_counts(y_test)}")
    print(f"  exact_train_test_overlap: {overlap_count}")
    print(f"  exact_feature_vector_overlap: {feature_overlap_count}")

    # The uncalibrated baseline is fit once for a calibration comparison only.
    # It is not used for hyperparameter selection or saved for production.
    baseline = build_base_pipeline()
    baseline.fit(X_train, y_train)
    baseline_probabilities = baseline.predict_proba(X_test)[:, 1]

    calibrated_model = build_calibrated_model()
    calibrated_model.fit(X_train, y_train)
    predictions = calibrated_model.predict(X_test)
    probabilities = calibrated_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(y_test, probabilities),
    }

    print("\nUntouched test evaluation")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"  confusion_matrix: {confusion_matrix(y_test, predictions).tolist()}")

    baseline_extreme_count = int(
        np.count_nonzero(
            (baseline_probabilities <= 0.05) | (baseline_probabilities >= 0.95)
        )
    )
    calibrated_extreme_count = int(
        np.count_nonzero((probabilities <= 0.05) | (probabilities >= 0.95))
    )
    print("\nCalibration comparison on untouched test set")
    print(
        "  uncalibrated: "
        f"log_loss={log_loss(y_test, baseline_probabilities, labels=[0, 1]):.6f}, "
        f"brier={brier_score_loss(y_test, baseline_probabilities):.6f}, "
        f"extreme_predictions={baseline_extreme_count}/{len(y_test)}, "
        f"range=[{baseline_probabilities.min():.6f}, "
        f"{baseline_probabilities.max():.6f}]"
    )
    print(
        "  calibrated: "
        f"log_loss={metrics['log_loss']:.6f}, "
        f"brier={metrics['brier_score']:.6f}, "
        f"extreme_predictions={calibrated_extreme_count}/{len(y_test)}, "
        f"range=[{probabilities.min():.6f}, {probabilities.max():.6f}]"
    )

    observed_fraction, predicted_mean = calibration_curve(
        y_test,
        probabilities,
        n_bins=5,
        strategy="quantile",
    )
    print("  calibration_bins:")
    for mean_probability, observed_rate in zip(
        predicted_mean,
        observed_fraction,
        strict=True,
    ):
        print(
            f"    mean_probability={mean_probability:.6f}, "
            f"observed_positive_rate={observed_rate:.6f}"
        )

    print("\nHeld-out row checks")
    held_out = X_test.copy()
    held_out["target"] = y_test
    held_out["prediction"] = predictions
    held_out["class_1_probability"] = probabilities
    for target in (0, 1):
        print(f"  target={target}")
        examples = held_out[held_out["target"] == target].sort_index().head(3)
        for index, row in examples.iterrows():
            vector = []
            for name in FEATURE_ORDER:
                value = X_test.at[index, name]
                vector.append(value.item() if isinstance(value, np.generic) else value)
            print(
                f"    source_index={index}, prediction={int(row['prediction'])}, "
                f"probability={row['class_1_probability']:.6f}, "
                f"vector={vector}"
            )

    reference_frame = pd.DataFrame([REFERENCE_PROFILE], columns=FEATURE_ORDER)
    legacy_model = joblib.load(LEGACY_MODEL_PATH)
    legacy_scaler = joblib.load(LEGACY_SCALER_PATH)
    legacy_scaled = legacy_scaler.transform(reference_frame)
    print("\nReference profile")
    print_reference_result("  legacy", legacy_model, legacy_scaled)
    print_reference_result("  calibrated", calibrated_model, reference_frame)

    joblib.dump(calibrated_model, ARTIFACT_PATH)
    print(f"\nSaved calibrated inference pipeline: {ARTIFACT_PATH.name}")


if __name__ == "__main__":
    main()

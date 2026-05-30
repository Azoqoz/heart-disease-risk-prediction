import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


# Load Dataset


df = pd.read_csv("heart.csv")


# Features and Target


X = df.drop("target", axis=1)
y = df["target"]


# Train Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Model


model = XGBClassifier()

model.fit(X_train_scaled, y_train)


# Predictions


y_pred = model.predict(X_test_scaled)


# Metrics


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# Save Model


joblib.dump(model, "heart_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("\nModel and scaler saved successfully!")
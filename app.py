from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import re

app = FastAPI()

# Load model
model = joblib.load("final_model.pkl")

# Derive feature names directly from the loaded model

def normalize_key(key: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower())
    return normalized.strip("_")


def best_feature_names(estimator, fallback=None):
    if hasattr(estimator, "feature_names_in_"):
        return list(estimator.feature_names_in_)

    def try_sub(est):
        if hasattr(est, "feature_names_in_"):
            return list(est.feature_names_in_)
        if hasattr(est, "base_estimator") and hasattr(est.base_estimator, "feature_names_in_"):
            return list(est.base_estimator.feature_names_in_)
        if hasattr(est, "estimator") and hasattr(est.estimator, "feature_names_in_"):
            return list(est.estimator.feature_names_in_)
        return None

    for attr in ("feature_names_in_", "calibrated_classifiers_", "base_estimator", "base_estimator_", "best_estimator_", "estimator", "estimator_", "classifiers_"):
        sub = getattr(estimator, attr, None)
        if sub is None:
            continue
        if isinstance(sub, list):
            for item in sub:
                names = try_sub(item)
                if names:
                    return names
        else:
            names = try_sub(sub)
            if names:
                return names

    return list(fallback) if fallback is not None else []


features = best_feature_names(model)
feature_name_map = {normalize_key(name): name for name in features}


@app.get("/")
def home():
    return {"message": "API is running"}


@app.get("/features")
def get_features():
    return {"features": list(features)}


@app.post("/predict")
def predict(data: dict):
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Input must be a JSON object.")

    normalized_input = {
        normalize_key(key): value for key, value in data.items()
    }

    row = {}
    for feature in features:
        row[feature] = normalized_input.get(normalize_key(feature))

    if "Pulse Pressure" in row and row.get("Pulse Pressure") is None:
        if row.get("Systolic BP") is None or row.get("Diastolic") is None:
            raise HTTPException(
                status_code=400,
                detail="Missing required values to compute Pulse Pressure. Provide Systolic BP and Diastolic."
            )
        row["Pulse Pressure"] = row["Systolic BP"] - row["Diastolic"]

    if "Mean Arterial Pressure" in row and row.get("Mean Arterial Pressure") is None:
        if row.get("Diastolic") is None or row.get("Pulse Pressure") is None:
            raise HTTPException(
                status_code=400,
                detail="Missing required values to compute Mean Arterial Pressure."
            )
        row["Mean Arterial Pressure"] = row["Diastolic"] + (row["Pulse Pressure"] / 3)

    missing = [feature for feature, value in row.items() if value is None]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing input values for features: {missing}"
        )

    # Create dataframe in exact order
    input_data = pd.DataFrame([row], columns=features)

    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0].max()

    # Label mapping
    risk_labels = {
        0: "Low Risk",
        1: "High Risk"
    }

    return {
        "prediction": risk_labels[int(prediction)],
        "confidence": float(probability)
    }
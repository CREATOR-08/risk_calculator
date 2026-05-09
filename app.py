from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("calibrated_model.pkl")

# Load feature names
features = joblib.load("feature_names.pkl")


@app.get("/")
def home():
    return {"message": "API is running"}


@app.get("/features")
def get_features():
    return {"features": list(features)}


@app.post("/predict")
def predict(data: dict):

    # Derived Features
    data["Pulse_Pressure"] = (
        data["Systolic_BP"] - data["Diastolic"]
    )

    data["Mean_Arterial_Pressure"] = (
        data["Diastolic"] +
        (data["Pulse_Pressure"] / 3)
    )

    # Create dataframe in exact order
    input_data = pd.DataFrame(
        [[data[col] for col in features]],
        columns=features
    )

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
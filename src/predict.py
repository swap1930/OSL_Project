import os

import joblib
import numpy as np
import pandas as pd
import shap

# Get absolute paths to model files
model_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"
)
model_path = os.path.join(model_dir, "model.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Feature names for explainability
feature_names = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Initialize SHAP explainer (only once)
explainer = shap.Explainer(
    model, pd.DataFrame([[300, 310, 1500, 40, 50]], columns=feature_names)
)


def predict(data):
    # Convert data to DataFrame with proper feature names
    data_df = pd.DataFrame([data], columns=feature_names)
    data_scaled = scaler.transform(data_df)
    result = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]  # Probability of failure
    return result, probability


def explain_prediction(data):
    """Explain which features contributed to the prediction"""
    # Convert data to DataFrame with proper feature names
    data_df = pd.DataFrame([data], columns=feature_names)
    data_scaled = scaler.transform(data_df)

    # Get SHAP values
    shap_values = explainer(data_scaled)

    # Create feature importance dictionary
    feature_importance = {}
    for i, feature in enumerate(feature_names):
        shap_value = shap_values.values[0][i]

        # Convert to numpy array if needed and get the first element
        if hasattr(shap_value, "flatten"):
            shap_value = np.array(shap_value).flatten()[0]
        elif hasattr(shap_value, "__len__") and len(shap_value) > 1:
            shap_value = np.array(shap_value)[0]

        # Now compare the scalar value
        impact_value = float(shap_value) > 0

        feature_importance[feature] = {
            "value": data[i],
            "shap_value": float(shap_value),
            "impact": "Increases Risk" if impact_value else "Decreases Risk",
        }

    # Sort by absolute SHAP value
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: abs(x[1]["shap_value"]), reverse=True
    )

    return sorted_features


def generate_realistic_data():
    """Generate realistic machine data with some patterns"""
    # Base values with some randomness
    air_temp = np.random.uniform(298, 305)  # K
    process_temp = np.random.uniform(308, 315)  # K (always higher than air)
    rpm = np.random.uniform(1400, 1600)  # rpm
    torque = np.random.uniform(30, 60)  # Nm
    wear = np.random.uniform(0, 200)  # min

    # Add some correlation between features
    if rpm > 1550:  # High RPM might increase temperature
        air_temp += np.random.uniform(0, 3)
        process_temp += np.random.uniform(0, 2)

    if torque > 50:  # High torque might increase temperature and wear
        air_temp += np.random.uniform(0, 2)
        process_temp += np.random.uniform(0, 1)
        wear += np.random.uniform(0, 10)

    return [air_temp, process_temp, rpm, torque, wear]

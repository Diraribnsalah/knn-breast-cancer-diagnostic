import os
import joblib
import pandas as pd
import numpy as np

#  Define project and artifact paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'knn_model.pkl')

#  Load scaler and model into memory upon file import
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

def predict_cancer(patient_features):
    """
    Accepts patient features (DataFrame, Dictionary, or NumPy Array),
    scales them, and returns the diagnosis with confidence percentages.
    """
    # Check input data structure and convert to DataFrame
    if isinstance(patient_features, dict):
        df_input = pd.DataFrame([patient_features])
    elif isinstance(patient_features, np.ndarray):
        df_input = patient_features
    else:
        df_input = patient_features

    # Scale features using the loaded scaler
    scaled_features = scaler.transform(df_input)

    # Predict and calculate probabilities
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]

    # Map output predictions
    diagnosis_map = {0: 'Malignant', 1: 'Benign'}

    result = {
        'Diagnosis': diagnosis_map[prediction],
        'Class_Code': int(prediction),
        'Confidence_Benign_%': round(float(probabilities[1]) * 100, 2),
        'Confidence_Malignant_%': round(float(probabilities[0]) * 100, 2)
    }
    return result

if __name__ == "__main__":
    print(" Inference Engine Ready.")

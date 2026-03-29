from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os

def preprocess(df):
    # Drop unnecessary columns
    df = df.drop(['UDI','Product ID','Type'], axis=1)
    
    # Only use the 5 features that are available in the UI
    feature_columns = [
        'Air temperature [K]',
        'Process temperature [K]', 
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    X = df[feature_columns]
    y = df['Machine failure']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler with absolute path
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    return X_scaled, y
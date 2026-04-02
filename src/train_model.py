from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)

    # Create model directory if it doesn't exist
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"
    )
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    return model

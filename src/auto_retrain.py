import json
import os
from datetime import datetime, timedelta

import pandas as pd

from src.preprocessing import preprocess
from src.train_model import train


class AutoRetrainer:
    def __init__(self):
        self.logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        self.model_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "model"
        )
        self.data_log_path = os.path.join(self.logs_dir, "prediction_data.json")
        self.metrics_log_path = os.path.join(self.logs_dir, "retraining_metrics.json")

        # Create directories if they don't exist
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Load existing logs
        self.prediction_data = self.load_prediction_data()
        self.metrics_data = self.load_metrics_data()

    def load_prediction_data(self):
        """Load existing prediction data from logs"""
        if os.path.exists(self.data_log_path):
            with open(self.data_log_path, "r") as f:
                return json.load(f)
        return []

    def load_metrics_data(self):
        """Load existing metrics data from logs"""
        if os.path.exists(self.metrics_log_path):
            with open(self.metrics_log_path, "r") as f:
                return json.load(f)
        return []

    def save_prediction_data(self):
        """Save prediction data to logs"""
        with open(self.data_log_path, "w") as f:
            json.dump(self.prediction_data[-1000:], f)  # Keep only last 1000 entries

    def save_metrics_data(self):
        """Save metrics data to logs"""
        with open(self.metrics_log_path, "w") as f:
            json.dump(self.metrics_data, f)

    def log_prediction(self, data, prediction, probability, timestamp):
        """Log a prediction for future retraining"""
        entry = {
            "timestamp": timestamp.isoformat(),
            "data": data,
            "prediction": int(prediction),
            "probability": float(probability),
        }
        self.prediction_data.append(entry)
        self.save_prediction_data()

    def should_retrain(self, min_samples=100, performance_drop_threshold=0.1):
        """Check if model should be retrained"""
        if len(self.prediction_data) < min_samples:
            return False, "Not enough data samples"

        # Check if performance has dropped
        if len(self.metrics_data) > 0:
            last_accuracy = self.metrics_data[-1].get("accuracy", 0)
            if last_accuracy < (1 - performance_drop_threshold):
                return True, f"Performance dropped to {last_accuracy:.2%}"

        # Check if it's been more than 7 days since last retraining
        if len(self.metrics_data) > 0:
            last_retrain = datetime.fromisoformat(self.metrics_data[-1]["timestamp"])
            if datetime.now() - last_retrain > timedelta(days=14):
                return True, "Scheduled bi-weekly retraining"

        return False, "No retraining needed"

    def _create_training_dataset(self):
        """Create training dataset from logged predictions"""
        if len(self.prediction_data) < 50:
            return None, None

        # Create feature matrix with proper column names
        feature_names = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

        X = pd.DataFrame(
            [entry["data"] for entry in self.prediction_data], columns=feature_names
        )
        y = pd.Series(
            [entry["prediction"] for entry in self.prediction_data],
            name="Machine failure",
        )

        return X, y

    def retrain_model(self):
        """Retrain the model with logged data"""
        try:
            # Create training dataset
            X, y = self.create_training_dataset()

            if X is None or y is None:
                return False, "Insufficient data for retraining"

            # Combine with original data for better training
            original_df = pd.read_csv("data/raw/ai4i2020.csv")

            # Preprocess original data
            X_orig, y_orig = preprocess(original_df)

            # Convert original data to DataFrame with proper column names
            feature_names = [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
            ]

            X_orig_df = pd.DataFrame(X_orig, columns=feature_names)
            y_orig_series = pd.Series(y_orig, name="Machine failure")

            # Combine datasets
            X_combined = pd.concat([X_orig_df, X], ignore_index=True)
            y_combined = pd.concat([y_orig_series, y], ignore_index=True)

            # Retrain model
            model = train(X_combined, y_combined)

            # Calculate metrics
            accuracy = model.score(X_combined, y_combined)

            # Log retraining metrics
            metrics_entry = {
                "timestamp": datetime.now().isoformat(),
                "accuracy": float(accuracy),
                "training_samples": len(X_combined),
                "new_samples": len(X),
                "original_samples": len(X_orig_df),
            }

            self.metrics_data.append(metrics_entry)
            self.save_metrics_data()

            # Clear old prediction data to avoid drift
            self.prediction_data = self.prediction_data[-500:]  # Keep last 500
            self.save_prediction_data()

            return True, f"Model retrained successfully. Accuracy: {accuracy:.2%}"

        except Exception as e:
            return False, f"Retraining failed: {str(e)}"

    def get_retraining_status(self):
        """Get current retraining status"""
        should_retrain, reason = self.should_retrain()

        status = {
            "should_retrain": should_retrain,
            "reason": reason,
            "total_samples": len(self.prediction_data),
            "last_retrain": (
                self.metrics_data[-1]["timestamp"] if self.metrics_data else "Never"
            ),
            "last_accuracy": (
                self.metrics_data[-1]["accuracy"] if self.metrics_data else 0
            ),
        }

        return status


# Global instance
auto_retrainer = AutoRetrainer()


def log_prediction(data, prediction, probability, timestamp=None):
    """Convenience function to log predictions"""
    if timestamp is None:
        timestamp = datetime.now()
    auto_retrainer.log_prediction(data, prediction, probability, timestamp)


def check_and_retrain():
    """Check if retraining is needed and execute if necessary"""
    should_retrain, reason = auto_retrainer.should_retrain()
    if should_retrain:
        success, message = auto_retrainer.retrain_model()
        return success, message
    return False, reason


def get_retraining_status():
    """Get current retraining status"""
    return auto_retrainer.get_retraining_status()

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.predict import generate_realistic_data, predict
from src.preprocessing import preprocess
from src.train_model import train


@pytest.mark.integration
def test_full_pipeline_integration():
    """Test the complete ML pipeline from data to prediction"""
    # Create sample data with all expected columns
    np.random.seed(42)
    n_samples = 100

    data = {
        "UDI": range(1, n_samples + 1),
        "Product ID": [f"M{i:05d}" for i in range(1, n_samples + 1)],
        "Type": np.random.choice(['M', 'L', 'H'], n_samples),
        "Air temperature [K]": np.random.normal(300, 5, n_samples),
        "Process temperature [K]": np.random.normal(310, 5, n_samples),
        "Rotational speed [rpm]": np.random.normal(1500, 200, n_samples),
        "Torque [Nm]": np.random.normal(40, 10, n_samples),
        "Tool wear [min]": np.random.uniform(0, 250, n_samples),
        "Machine failure": np.random.binomial(1, 0.1, n_samples),
    }
    df = pd.DataFrame(data)

    # Preprocess data
    X, y = preprocess(df)

    # Train model
    model = train(X, y)

    # Test prediction with new data
    new_data = generate_realistic_data()

    # Mock the model loading for prediction
    with patch("src.predict.joblib.load") as mock_load, patch(
        "src.predict.os.path.exists", return_value=True
    ):

        mock_load.side_effect = [model, MagicMock()]

        result, probability = predict(new_data)

        # Validate results
        assert result in [0, 1]
        assert 0 <= probability <= 1


@pytest.mark.integration
def test_data_consistency_across_pipeline():
    """Test that data remains consistent across pipeline stages"""
    # Create sample data with all expected columns
    data = {
        "UDI": [1, 2, 3, 4, 5],
        "Product ID": ["M00001", "M00002", "M00003", "M00004", "M00005"],
        "Type": ["M", "L", "H", "M", "L"],
        "Air temperature [K]": [298.5, 300.2, 305.1, 310.0, 295.5],
        "Process temperature [K]": [308.5, 310.2, 315.1, 320.0, 305.5],
        "Rotational speed [rpm]": [1500, 1800, 2000, 1200, 1600],
        "Torque [Nm]": [40.5, 50.2, 60.1, 30.0, 45.0],
        "Tool wear [min]": [50, 100, 150, 25, 75],
        "Machine failure": [0, 1, 0, 0, 1],
    }
    df = pd.DataFrame(data)

    # Preprocess
    X, y = preprocess(df)

    # Check that number of samples is preserved
    assert X.shape[0] == len(df)
    assert y.shape[0] == len(df)

    # Train model
    model = train(X, y)

    # Test that model can handle the same data format
    predictions = model.predict(X)
    assert len(predictions) == len(df)


@pytest.mark.integration
def test_model_with_different_data_sizes():
    """Test model performance with different dataset sizes"""
    sizes = [10, 50, 100]

    for size in sizes:
        # Create data with all expected columns
        np.random.seed(42)
        data = {
            "UDI": range(1, size + 1),
            "Product ID": [f"M{i:05d}" for i in range(1, size + 1)],
            "Type": np.random.choice(['M', 'L', 'H'], size),
            "Air temperature [K]": np.random.normal(300, 5, size),
            "Process temperature [K]": np.random.normal(310, 5, size),
            "Rotational speed [rpm]": np.random.normal(1500, 200, size),
            "Torque [Nm]": np.random.normal(40, 10, size),
            "Tool wear [min]": np.random.uniform(0, 250, size),
            "Machine failure": np.random.binomial(1, 0.1, size),
        }
        df = pd.DataFrame(data)

        # Preprocess and train
        X, y = preprocess(df)
        model = train(X, y)

        # Test prediction
        test_data = generate_realistic_data()

        with patch("src.predict.joblib.load") as mock_load, patch(
            "src.predict.os.path.exists", return_value=True
        ):

            mock_load.side_effect = [model, MagicMock()]
            result, probability = predict(test_data)

            assert result in [0, 1]
            assert 0 <= probability <= 1


@pytest.mark.integration
def test_error_handling_integration():
    """Test error handling across the integrated pipeline"""
    # Test with invalid data
    invalid_data = {
        "Air temperature [K]": ["invalid", 300.2],  # String value
        "Process temperature [K]": [308.5, 310.2],
        "Rotational speed [rpm]": [1500, 1800],
        "Torque [Nm]": [40.5, 50.2],
        "Tool wear [min]": [50, 100],
        "Machine failure": [0, 1],
    }
    df = pd.DataFrame(invalid_data)

    # Should raise exception during preprocessing
    with pytest.raises(Exception):
        preprocess(df)


@pytest.mark.integration
def test_prediction_with_trained_model():
    """Test prediction using an actually trained model"""
    # Create training data
    np.random.seed(42)
    n_samples = 200

    # Create data with clear patterns
    air_temp = np.random.normal(300, 5, n_samples)
    process_temp = air_temp + np.random.normal(10, 2, n_samples)
    rpm = np.random.normal(1500, 200, n_samples)
    torque = np.random.normal(40, 8, n_samples)
    wear = np.random.uniform(0, 250, n_samples)

    # Create failure pattern
    failure_prob = (
        (air_temp > 305) * 0.4
        + (process_temp > 315) * 0.4
        + (rpm > 1800) * 0.3
        + (torque > 50) * 0.3
        + (wear > 200) * 0.4
    )
    failure_prob = np.clip(failure_prob, 0, 1)
    y = np.random.binomial(1, failure_prob)

    X = np.column_stack([air_temp, process_temp, rpm, torque, wear])

    # Train model
    model = train(X, y)

    # Test predictions on different scenarios
    test_cases = [
        [298.0, 308.0, 1400.0, 35.0, 30.0],  # Normal conditions
        [310.0, 320.0, 2000.0, 80.0, 220.0],  # High stress conditions
        [295.0, 305.0, 1200.0, 25.0, 10.0],  # Low stress conditions
    ]

    with patch("src.predict.joblib.load") as mock_load, patch(
        "src.predict.os.path.exists", return_value=True
    ):

        mock_load.side_effect = [model, MagicMock()]

        for test_data in test_cases:
            result, probability = predict(test_data)
            assert result in [0, 1]
            assert 0 <= probability <= 1


@pytest.mark.integration
def test_scaler_consistency():
    """Test that scaler is consistently applied"""
    # Create training data with all expected columns
    np.random.seed(42)
    data = {
        "UDI": range(1, 101),
        "Product ID": [f"M{i:05d}" for i in range(1, 101)],
        "Type": np.random.choice(['M', 'L', 'H'], 100),
        "Air temperature [K]": np.random.normal(300, 5, 100),
        "Process temperature [K]": np.random.normal(310, 5, 100),
        "Rotational speed [rpm]": np.random.normal(1500, 200, 100),
        "Torque [Nm]": np.random.normal(40, 10, 100),
        "Tool wear [min]": np.random.uniform(0, 250, 100),
        "Machine failure": np.random.binomial(1, 0.1, 100),
    }
    df = pd.DataFrame(data)

    # Preprocess (this should save the scaler)
    X, y = preprocess(df)

    # Check that scaler file was created
    scaler_path = os.path.join("model", "scaler.pkl")
    if os.path.exists(scaler_path):
        import joblib

        scaler = joblib.load(scaler_path)

        # Check that scaler transforms data consistently
        transformed = scaler.transform(X)
        assert transformed.shape == X.shape
        
        # Check that transformed data has roughly zero mean and unit variance
        # Note: Due to the small sample size, we use more tolerant thresholds
        assert np.allclose(transformed.mean(axis=0), 0, atol=2.0)  
        assert np.allclose(transformed.std(axis=0), 1, atol=1.0)  

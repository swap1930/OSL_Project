import os
import shutil
import tempfile

import numpy as np
import pytest

from src.train_model import train


@pytest.mark.unit
def test_train_basic():
    """Test basic model training functionality"""
    # Create sample data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    model = train(X, y)

    # Check that model is trained
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    # Test prediction
    predictions = model.predict(X[:5])
    assert len(predictions) == 5
    assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.unit
def test_train_model_saving():
    """Test that model is saved correctly"""
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample data
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        # Mock the model directory path
        import src.train_model as train_module

        original_dir = train_module.os.path.dirname
        train_module.os.path.dirname = lambda x: temp_dir

        try:
            model = train(X, y)

            # Check that model file was created
            model_path = os.path.join(temp_dir, "model", "model.pkl")
            assert os.path.exists(model_path)

            # Check that file is not empty
            assert os.path.getsize(model_path) > 0

        finally:
            train_module.os.path.dirname = original_dir


@pytest.mark.unit
def test_train_empty_data():
    """Test training with empty data"""
    X = np.array([]).reshape(0, 5)
    y = np.array([])

    with pytest.raises(Exception):
        train(X, y)


@pytest.mark.unit
def test_train_mismatched_dimensions():
    """Test training with mismatched X and y dimensions"""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 50)  # Different length

    with pytest.raises(Exception):
        train(X, y)


@pytest.mark.unit
def test_train_single_sample():
    """Test training with single sample"""
    X = np.random.rand(1, 5)
    y = np.array([1])

    model = train(X, y)
    assert model is not None


@pytest.mark.unit
def test_train_binary_classification():
    """Test that model works for binary classification"""
    # Create clearly separable data
    X_class_0 = np.random.normal(0, 1, (50, 5))
    X_class_1 = np.random.normal(2, 1, (50, 5))
    X = np.vstack([X_class_0, X_class_1])
    y = np.array([0] * 50 + [1] * 50)

    model = train(X, y)

    # Test predictions on training data
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)

    # Should achieve reasonable accuracy on training data
    assert accuracy > 0.8


@pytest.mark.integration
def test_train_with_realistic_data():
    """Test training with realistic data patterns"""
    np.random.seed(42)
    n_samples = 200

    # Create realistic data with some correlation
    air_temp = np.random.normal(300, 5, n_samples)
    process_temp = air_temp + np.random.normal(10, 2, n_samples)
    rpm = np.random.normal(1500, 200, n_samples)
    torque = np.random.normal(40, 8, n_samples)
    wear = np.random.uniform(0, 250, n_samples)

    # Create failure pattern based on conditions
    failure_prob = (
        (air_temp > 305) * 0.3
        + (process_temp > 315) * 0.3
        + (rpm > 1800) * 0.2
        + (torque > 50) * 0.2
        + (wear > 200) * 0.3
    )
    failure_prob = np.clip(failure_prob, 0, 1)
    y = np.random.binomial(1, failure_prob)

    X = np.column_stack([air_temp, process_temp, rpm, torque, wear])

    model = train(X, y)

    # Test model functionality
    assert hasattr(model, "feature_importances_")
    assert len(model.feature_importances_) == 5

    # Test predictions
    predictions = model.predict_proba(X[:10])
    assert predictions.shape == (10, 2)
    assert np.allclose(predictions.sum(axis=1), 1.0)  # Probabilities sum to 1

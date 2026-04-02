import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.predict import explain_prediction, generate_realistic_data, predict


@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.8, 0.2]])
    return model


@pytest.fixture
def mock_scaler():
    """Create a mock scaler for testing"""
    scaler = MagicMock()
    scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    return scaler


@pytest.mark.unit
@patch("src.predict.joblib.load")
@patch("src.predict.os.path.exists")
def test_predict_basic(mock_exists, mock_load, mock_model, mock_scaler):
    """Test basic prediction functionality"""
    # Mock file existence and loading
    mock_exists.return_value = True
    mock_load.side_effect = [mock_model, mock_scaler]

    # Test prediction
    data = [300.0, 310.0, 1500.0, 40.0, 50.0]
    result, probability = predict(data)

    # Check results
    assert result == 0
    assert probability == 0.2  # Probability of class 1
    assert isinstance(result, int)
    assert isinstance(probability, float)
    assert 0 <= probability <= 1


@pytest.mark.unit
@patch("src.predict.joblib.load")
@patch("src.predict.os.path.exists")
def test_predict_failure_case(mock_exists, mock_load):
    """Test prediction with failure case"""
    # Mock model that predicts failure
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])

    scaler = MagicMock()
    scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

    mock_exists.return_value = True
    mock_load.side_effect = [model, scaler]

    data = [310.0, 320.0, 2000.0, 80.0, 200.0]
    result, probability = predict(data)

    assert result == 1
    assert probability == 0.7


@pytest.mark.unit
@patch("src.predict.os.path.exists")
def test_predict_model_not_found(mock_exists):
    """Test prediction when model files don't exist"""
    mock_exists.return_value = False

    data = [300.0, 310.0, 1500.0, 40.0, 50.0]

    with pytest.raises(Exception):
        predict(data)


@pytest.mark.unit
def test_predict_invalid_input_length():
    """Test prediction with invalid input length"""
    with patch("src.predict.os.path.exists", return_value=True), patch(
        "src.predict.joblib.load"
    ) as mock_load:

        mock_load.return_value = MagicMock()

        # Test with wrong number of features
        data = [300.0, 310.0]  # Only 2 features instead of 5

        with pytest.raises(Exception):
            predict(data)


@pytest.mark.unit
def test_generate_realistic_data():
    """Test realistic data generation"""
    data = generate_realistic_data()

    # Check data structure
    assert isinstance(data, list)
    assert len(data) == 5

    # Check data ranges (reasonable values for machine parameters)
    assert 250 <= data[0] <= 350  # Air temperature (K)
    assert 250 <= data[1] <= 400  # Process temperature (K)
    assert 1000 <= data[2] <= 3000  # RPM
    assert 0 <= data[3] <= 100  # Torque (Nm)
    assert 0 <= data[4] <= 300  # Tool wear (min)

    # Check data types
    assert all(isinstance(x, (int, float)) for x in data)


@pytest.mark.unit
def test_generate_realistic_data_variability():
    """Test that generated data has variability"""
    # Generate multiple samples
    samples = [generate_realistic_data() for _ in range(100)]

    # Convert to numpy array for analysis
    samples_array = np.array(samples)

    # Check that there's variability in each feature
    for i in range(5):
        assert samples_array[:, i].std() > 0, f"Feature {i} has no variability"


@pytest.mark.unit
@patch("src.predict.joblib.load")
@patch("src.predict.os.path.exists")
def test_explain_prediction(mock_exists, mock_load, mock_model, mock_scaler):
    """Test prediction explanation functionality"""
    mock_exists.return_value = True
    mock_load.side_effect = [mock_model, mock_scaler]

    data = [300.0, 310.0, 1500.0, 40.0, 50.0]

    # Test explanation (assuming explain_prediction returns some explanation)
    try:
        explanation = explain_prediction(data)
        assert explanation is not None
        assert isinstance(explanation, (str, dict))
    except ImportError:
        # SHAP might not be installed in test environment
        pytest.skip("SHAP not available for testing")


@pytest.mark.integration
def test_predict_with_real_data_flow():
    """Test prediction with real data flow"""
    # This test requires actual model files
    # Skip if model files don't exist
    if not os.path.exists("model/model.pkl") or not os.path.exists("model/scaler.pkl"):
        pytest.skip("Model files not found")

    data = [300.0, 310.0, 1500.0, 40.0, 50.0]

    try:
        result, probability = predict(data)
        assert result in [0, 1]
        assert 0 <= probability <= 1
    except Exception as e:
        pytest.fail(f"Prediction failed with real model: {e}")


@pytest.mark.unit
def test_predict_edge_cases():
    """Test prediction with edge case values"""
    with patch("src.predict.os.path.exists", return_value=True), patch(
        "src.predict.joblib.load"
    ) as mock_load:

        # Mock model and scaler
        model = MagicMock()
        model.predict.return_value = np.array([0])
        model.predict_proba.return_value = np.array([[0.5, 0.5]])

        scaler = MagicMock()
        scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        mock_load.side_effect = [model, scaler]

        # Test with minimum values
        data = [250.0, 250.0, 1000.0, 0.0, 0.0]
        result, probability = predict(data)
        assert result in [0, 1]
        assert 0 <= probability <= 1

        # Test with maximum values
        data = [350.0, 400.0, 3000.0, 100.0, 300.0]
        result, probability = predict(data)
        assert result in [0, 1]
        assert 0 <= probability <= 1


@pytest.mark.slow
def test_predict_performance():
    """Test prediction performance"""
    with patch("src.predict.os.path.exists", return_value=True), patch(
        "src.predict.joblib.load"
    ) as mock_load:

        # Mock model and scaler
        model = MagicMock()
        model.predict.return_value = np.array([0])
        model.predict_proba.return_value = np.array([[0.8, 0.2]])

        scaler = MagicMock()
        scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        mock_load.side_effect = [model, scaler]

        import time

        data = [300.0, 310.0, 1500.0, 40.0, 50.0]

        # Test multiple predictions
        start_time = time.time()
        for _ in range(100):
            predict(data)
        end_time = time.time()

        # Should complete 100 predictions in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0

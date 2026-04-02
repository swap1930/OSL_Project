import pytest
import pandas as pd
import numpy as np
from src.preprocessing import preprocess


@pytest.mark.unit
def test_preprocess_basic():
    """Test basic preprocessing functionality"""
    # Create sample data
    data = {
        'Air temperature [K]': [298.5, 300.2, 305.1],
        'Process temperature [K]': [308.5, 310.2, 315.1],
        'Rotational speed [rpm]': [1500, 1800, 2000],
        'Torque [Nm]': [40.5, 50.2, 60.1],
        'Tool wear [min]': [50, 100, 150],
        'Machine failure': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    X, y = preprocess(df)
    
    # Check output shapes
    assert X.shape[0] == 3  # Same number of samples
    assert y.shape[0] == 3  # Same number of targets
    assert X.shape[1] == 5  # 5 features
    assert len(y) == 3
    
    # Check data types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.dtype == np.float64
    assert y.dtype in [np.int32, np.int64]


@pytest.mark.unit
def test_preprocess_empty_dataframe():
    """Test preprocessing with empty dataframe"""
    df = pd.DataFrame()
    
    with pytest.raises(Exception):
        preprocess(df)


@pytest.mark.unit
def test_preprocess_missing_columns():
    """Test preprocessing with missing required columns"""
    data = {
        'Air temperature [K]': [298.5, 300.2],
        'Process temperature [K]': [308.5, 310.2],
        # Missing other required columns
    }
    df = pd.DataFrame(data)
    
    with pytest.raises(Exception):
        preprocess(df)


@pytest.mark.unit
def test_preprocess_single_sample():
    """Test preprocessing with single sample"""
    data = {
        'Air temperature [K]': [298.5],
        'Process temperature [K]': [308.5],
        'Rotational speed [rpm]': [1500],
        'Torque [Nm]': [40.5],
        'Tool wear [min]': [50],
        'Machine failure': [0]
    }
    df = pd.DataFrame(data)
    
    X, y = preprocess(df)
    
    assert X.shape == (1, 5)
    assert y.shape == (1,)


@pytest.mark.integration
def test_preprocess_real_data_structure():
    """Test preprocessing with data structure similar to real dataset"""
    # Simulate real data structure
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Air temperature [K]': np.random.normal(300, 2, n_samples),
        'Process temperature [K]': np.random.normal(310, 3, n_samples),
        'Rotational speed [rpm]': np.random.normal(1500, 100, n_samples),
        'Torque [Nm]': np.random.normal(40, 10, n_samples),
        'Tool wear [min]': np.random.uniform(0, 300, n_samples),
        'Machine failure': np.random.binomial(1, 0.1, n_samples)
    }
    df = pd.DataFrame(data)
    
    X, y = preprocess(df)
    
    # Check that preprocessing doesn't change sample count
    assert X.shape[0] == n_samples
    assert y.shape[0] == n_samples
    
    # Check that features are scaled (roughly)
    assert np.abs(X.mean()) < 1.0  # Should be close to 0 after scaling
    assert np.abs(X.std() - 1.0) < 0.5  # Should be close to 1 after scaling

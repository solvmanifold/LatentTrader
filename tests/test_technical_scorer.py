"""Tests for the technical scoring model."""

import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

from trading_advisor.models.technical_scorer import TechnicalScorer
from trading_advisor.analysis import calculate_technical_indicators

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Generate at least 60 days of sample price data to satisfy technical indicator requirements
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    data = {
        "date": dates,
        "Open": np.linspace(100, 120, 60) + np.random.normal(0, 1, 60),
        "High": np.linspace(101, 121, 60) + np.random.normal(0, 1, 60),
        "Low": np.linspace(99, 119, 60) + np.random.normal(0, 1, 60),
        "Close": np.linspace(100, 120, 60) + np.random.normal(0, 1, 60),
        "Volume": np.random.randint(1000, 2000, 60),
        "analyst_targets": np.random.uniform(110, 130, 60),
    }
    df = pd.DataFrame(data)
    df = calculate_technical_indicators(df)
    return df

@pytest.fixture
def model():
    """Create a model instance for testing."""
    return TechnicalScorer()

def test_model_initialization(model):
    """Test model initialization."""
    assert model.model_name == "TechnicalScorer"
    assert len(model.feature_columns) > 0
    assert len(model.scoring_components) > 0

def test_prepare_input(model, sample_data):
    """Test input preparation."""
    # Drop all rows with any NaNs before passing to prepare_input
    valid_data = sample_data.dropna()
    x = model.prepare_input(valid_data)
    assert isinstance(x, torch.Tensor)
    assert x.shape[1] == len(model.feature_columns)
    assert not torch.isnan(x).any()

def test_forward_pass(model, sample_data):
    """Test forward pass of the model."""
    x = model.prepare_input(sample_data)
    outputs = model.forward(x)
    
    assert 'total_score' in outputs
    assert 'details' in outputs
    assert isinstance(outputs['total_score'], torch.Tensor)
    assert isinstance(outputs['details'], dict)
    
    # Check score range
    assert torch.all(outputs['total_score'] >= 0)
    assert torch.all(outputs['total_score'] <= 10)
    
    # Check details
    for component, details in outputs['details'].items():
        assert isinstance(details, torch.Tensor)
        assert details.shape == outputs['total_score'].shape

def test_predict(model, sample_data):
    """Test prediction method."""
    predictions = model.predict(sample_data)
    
    assert 'score' in predictions
    assert 'details' in predictions
    assert isinstance(predictions['score'], np.ndarray)
    assert isinstance(predictions['details'], dict)
    
    # Check score range
    assert np.all(predictions['score'] >= 0)
    assert np.all(predictions['score'] <= 10)
    
    # Check details
    for component, details in predictions['details'].items():
        assert isinstance(details, np.ndarray)
        assert details.shape == predictions['score'].shape

def test_missing_columns(model):
    """Test handling of missing columns."""
    df = pd.DataFrame({'Close': [100.0]})
    with pytest.raises(ValueError):
        model.prepare_input(df)

def test_save_load(model, tmp_path):
    """Test model saving and loading."""
    # Save model
    save_path = tmp_path / "model.pt"
    model.save(str(save_path))
    assert save_path.exists()
    
    # Load model
    loaded_model = TechnicalScorer.load(str(save_path))
    assert isinstance(loaded_model, TechnicalScorer)
    assert loaded_model.model_name == model.model_name

def test_scoring_components(model, sample_data):
    """Test individual scoring components."""
    valid_data = sample_data.dropna()
    x = model.prepare_input(valid_data)
    
    if len(valid_data) == 0:
        pytest.skip('No valid rows after dropping NaNs')
    
    # RSI
    rsi = torch.tensor(valid_data['RSI'].values, dtype=torch.float32)
    rsi_score, _ = model._score_rsi(rsi)
    assert torch.all((rsi_score >= -1) & (rsi_score <= 1))
    
    # Bollinger Bands
    bb_pband = torch.tensor(valid_data['BB_Pband'].values, dtype=torch.float32)
    bb_score, _ = model._score_bollinger(bb_pband)
    assert torch.all((bb_score >= -2) & (bb_score <= 2))
    
    # MACD
    macd = torch.tensor(valid_data['MACD'].values, dtype=torch.float32)
    macd_signal = torch.tensor(valid_data['MACD_Signal'].values, dtype=torch.float32)
    macd_hist = torch.tensor(valid_data['MACD_Hist'].values, dtype=torch.float32)
    macd_score, _ = model._score_macd(macd, macd_signal, macd_hist)
    assert torch.all((macd_score >= -2) & (macd_score <= 2))
    
    # Moving Averages
    close = torch.tensor(valid_data['Close'].values, dtype=torch.float32)
    sma20 = torch.tensor(valid_data['SMA_20'].values, dtype=torch.float32)
    sma50 = torch.tensor(valid_data['SMA_50'].values, dtype=torch.float32)
    ma_score, _ = model._score_moving_averages(close, sma20, sma50)
    assert torch.all((ma_score >= -2) & (ma_score <= 2))
    
    # Volume
    volume = torch.tensor(valid_data['Volume'].values, dtype=torch.float32)
    prev_volume = torch.roll(volume, 1)
    prev_volume[0] = volume[0]
    volume_score, _ = model._score_volume(volume, prev_volume)
    assert torch.all((volume_score >= -1) & (volume_score <= 1)) 
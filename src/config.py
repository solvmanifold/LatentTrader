"""Configuration settings for the Weekly Trading Advisor."""

from pathlib import Path

# Directory paths
DATA_DIR = Path("data")

# Analysis parameters
LOOKBACK_DAYS = 100
REQUIRED_COLUMNS = ["Close", "High", "Low", "Open", "Volume"]

# MACD thresholds
MACD_STRONG_DIVERGENCE = 2.0  # Histogram value for strong divergence
MACD_WEAK_DIVERGENCE = 0.5    # Histogram value for weak divergence

# Scoring parameters
SCORE_WEIGHTS = {
    'rsi_oversold': 2,      # RSI below 30
    'rsi_overbought': -1,   # RSI above 70 (penalty)
    'bollinger': 2,         # Price outside Bollinger Bands
    'macd_crossover': 2,    # Strong bearish MACD
    'macd_strong_divergence': 2,  # Strong bullish MACD
    'macd_moderate_divergence': 1,  # Weak bullish MACD
    'sma_strong_above': 2,  # Price > 2% above 20-day MA
    'sma_strong_below': 2,  # Price < 2% below 20-day MA
    'sma_above': 1,         # Price above 20-day MA
    'volume_spike': 1,      # Volume spike > 20%
    'analyst_high_upside': 2,  # >20% upside to target
    'analyst_moderate_upside': 1  # 10-20% upside to target
}

# Maximum possible raw score before normalization (reflects actual logic, not just sum of weights)
MAX_RAW_SCORE = sum([
    2.0,  # RSI oversold
    2.0,  # Bollinger lower
    2.0,  # MACD strong divergence
    2.0,  # SMA > 2% above
    1.0,  # Volume spike
    2.0   # Analyst >20%
])  # = 11.0 
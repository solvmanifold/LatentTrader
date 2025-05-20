"""Configuration settings for the Weekly Trading Advisor."""

from pathlib import Path

# Directory paths
DATA_DIR = Path("data")

# Analysis parameters
LOOKBACK_DAYS = 100
REQUIRED_COLUMNS = ["Close", "High", "Low", "Open", "Volume"]

# Scoring parameters
SCORE_WEIGHTS = {
    'rsi_oversold': 2,
    'rsi_overbought': 1,
    'bb_upper': 2,
    'bb_lower': 2,
    'macd_crossover': 2,
    'macd_strong_divergence': 2,
    'macd_moderate_divergence': 1,
    'macd_acceleration': 1,
    'volume_spike': 1,
    'analyst_high_upside': 2,
    'analyst_moderate_upside': 1
}

# Maximum possible raw score before normalization
MAX_RAW_SCORE = sum(SCORE_WEIGHTS.values()) 
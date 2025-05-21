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
    'rsi': 2.0,
    'bollinger': 2.0,
    'macd': 2.0,
    'sma_above': 2.0,
    'analyst_targets': 2.0
}

# Maximum possible raw score before normalization
MAX_RAW_SCORE = sum(SCORE_WEIGHTS.values()) 
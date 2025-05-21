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
    'rsi_overbought': 1,    # RSI above 70
    'bb_upper': 2,          # Price above upper band
    'bb_lower': 2,          # Price below lower band
    'macd_crossover': 2,    # Strong bearish MACD
    'macd_strong_divergence': 2,  # Strong bullish MACD
    'macd_moderate_divergence': 1,  # Weak bullish MACD
    'sma_above': 1,         # Price above 20-day MA
    'volume_spike': 1,      # Volume spike > 20%
    'analyst_high_upside': 2,  # >20% upside to target
    'analyst_moderate_upside': 1  # 10-20% upside to target
}

# Maximum possible raw score before normalization
MAX_RAW_SCORE = sum(SCORE_WEIGHTS.values()) 
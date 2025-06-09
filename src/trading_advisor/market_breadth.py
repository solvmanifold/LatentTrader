"""Market breadth indicators calculation."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from trading_advisor.data import load_tickers, normalize_ticker, fill_missing_trading_days
from trading_advisor.features import load_features

logger = logging.getLogger(__name__)

def calculate_market_breadth(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market breadth indicators.
    
    Args:
        combined_df: DataFrame with combined ticker data
        
    Returns:
        DataFrame with market breadth indicators
    """
    if combined_df.empty:
        logger.warning("No data provided for market breadth calculation")
        return pd.DataFrame()
        
    # Calculate market breadth indicators
    breadth_df = pd.DataFrame()
    
    # Advance/Decline Line
    breadth_df['daily_breadth_adv_dec_line'] = combined_df.groupby(level=0)['close'].apply(
        lambda x: (x > x.shift(1)).sum() - (x < x.shift(1)).sum()
    )
    
    # New Highs/Lows
    breadth_df['daily_breadth_new_highs'] = combined_df.groupby(level=0)['close'].apply(
        lambda x: (x > x.rolling(20).max().shift(1)).sum()
    )
    breadth_df['daily_breadth_new_lows'] = combined_df.groupby(level=0)['close'].apply(
        lambda x: (x < x.rolling(20).min().shift(1)).sum()
    )
    
    # Moving Average Indicators
    breadth_df['daily_breadth_above_ma20'] = combined_df.groupby(level=0)['close'].apply(
        lambda x: (x > x.rolling(20).mean()).mean() * 100
    )
    breadth_df['daily_breadth_above_ma50'] = combined_df.groupby(level=0)['close'].apply(
        lambda x: (x > x.rolling(50).mean()).mean() * 100
    )
    
    # RSI Indicators
    breadth_df['daily_breadth_rsi_bullish'] = combined_df.groupby(level=0)['rsi'].apply(
        lambda x: (x > 50).mean() * 100
    )
    breadth_df['daily_breadth_rsi_oversold'] = combined_df.groupby(level=0)['rsi'].apply(
        lambda x: (x < 30).mean() * 100
    )
    breadth_df['daily_breadth_rsi_overbought'] = combined_df.groupby(level=0)['rsi'].apply(
        lambda x: (x > 70).mean() * 100
    )
    
    # MACD Indicators
    breadth_df['daily_breadth_macd_bullish'] = combined_df.groupby(level=0)['macd'].apply(
        lambda x: (x > 0).mean() * 100
    )
    
    # Fill in missing trading days
    breadth_df = fill_missing_trading_days(breadth_df, combined_df)
    
    return breadth_df 
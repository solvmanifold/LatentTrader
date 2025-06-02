"""Market breadth indicators calculation."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from trading_advisor.data import load_tickers
from trading_advisor.features import load_features

logger = logging.getLogger(__name__)

def calculate_market_breadth(
    tickers: Optional[List[str]] = None,
    features_dir: str = "features",
    output_dir: str = "market_features/breadth"
) -> pd.DataFrame:
    """
    Calculate market breadth indicators for a list of tickers.
    
    Args:
        tickers: List of tickers to analyze. If None, uses all tickers in features_dir
        features_dir: Directory containing feature files
        output_dir: Directory to save market breadth data
        
    Returns:
        DataFrame with market breadth indicators, one row per date
    """
    # Load tickers if not provided
    if tickers is None:
        tickers = [f.stem.replace('_features', '') for f in Path(features_dir).glob('*_features.parquet')]
    
    # Initialize results DataFrame
    dates = None
    breadth_data = []
    
    # Process each ticker
    for ticker in tqdm(tickers, desc="Calculating market breadth"):
        try:
            # Load features for this ticker
            df = load_features(ticker, features_dir)
            if df.empty:
                continue
                
            # Get dates if not set
            if dates is None:
                dates = df.index
                
            # Calculate indicators for this ticker
            ticker_data = pd.DataFrame(index=dates)
            
            # Moving averages
            ticker_data['above_ma20'] = df['Close'] > df['MA20']
            ticker_data['above_ma50'] = df['Close'] > df['MA50']
            ticker_data['above_ma200'] = df['Close'] > df['MA200']
            
            # RSI conditions
            ticker_data['oversold'] = df['RSI'] < 30
            ticker_data['overbought'] = df['RSI'] > 70
            
            # MACD crossovers
            ticker_data['macd_bullish'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
            ticker_data['macd_bearish'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
            
            # Volume trends
            ticker_data['volume_above_avg'] = df['Volume'] > df['Volume_MA20']
            
            breadth_data.append(ticker_data)
            
        except Exception as e:
            logger.warning(f"Failed to process {ticker}: {e}")
            continue
    
    if not breadth_data:
        logger.error("No valid data found for any tickers")
        return pd.DataFrame()
    
    # Combine all ticker data
    combined = pd.concat(breadth_data, axis=1)
    
    # Calculate market-wide metrics
    market_breadth = pd.DataFrame(index=dates)
    
    # Moving average breadth
    market_breadth['pct_above_ma20'] = combined['above_ma20'].mean(axis=1) * 100
    market_breadth['pct_above_ma50'] = combined['above_ma50'].mean(axis=1) * 100
    market_breadth['pct_above_ma200'] = combined['above_ma200'].mean(axis=1) * 100
    
    # RSI breadth
    market_breadth['pct_oversold'] = combined['oversold'].mean(axis=1) * 100
    market_breadth['pct_overbought'] = combined['overbought'].mean(axis=1) * 100
    
    # MACD breadth
    market_breadth['pct_macd_bullish'] = combined['macd_bullish'].mean(axis=1) * 100
    market_breadth['pct_macd_bearish'] = combined['macd_bearish'].mean(axis=1) * 100
    
    # Volume breadth
    market_breadth['pct_volume_above_avg'] = combined['volume_above_avg'].mean(axis=1) * 100
    
    # Save to parquet
    output_path = Path(output_dir) / "daily_breadth.parquet"
    output_path.parent.mkdir(exist_ok=True)
    market_breadth.to_parquet(output_path)
    
    logger.info(f"Saved market breadth data to {output_path}")
    return market_breadth 
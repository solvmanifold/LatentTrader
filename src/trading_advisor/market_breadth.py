"""Market breadth indicators calculation."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from trading_advisor.data import load_tickers, normalize_ticker
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
    
    # Required columns
    required_columns = {
        'price': ['Close'],
        'moving_averages': ['SMA_20', 'SMA_50', 'SMA_200'],
        'rsi': ['RSI'],
        'macd': ['MACD', 'MACD_Signal'],
        'volume': ['Volume', 'Volume_Prev'],
        'bollinger': ['BB_Upper', 'BB_Lower', 'BB_Middle']
    }
    
    # Process each ticker
    for ticker in tqdm(tickers, desc="Calculating market breadth"):
        try:
            # Normalize ticker name
            norm_ticker = normalize_ticker(ticker)
            
            # Load features for this ticker
            df = load_features(norm_ticker, features_dir)
            if df.empty:
                continue
                
            # Check for required columns
            missing_columns = []
            for category, cols in required_columns.items():
                for col in cols:
                    if col not in df.columns:
                        missing_columns.append(col)
            
            if missing_columns:
                logger.warning(f"Skipping {ticker}: Missing columns {missing_columns}")
                continue
                
            # Get dates if not set
            if dates is None:
                dates = df.index
                
            # Calculate indicators for this ticker
            ticker_data = pd.DataFrame(index=dates)
            
            # Moving averages
            ticker_data['above_sma20'] = df['Close'] > df['SMA_20']
            ticker_data['above_sma50'] = df['Close'] > df['SMA_50']
            ticker_data['above_sma200'] = df['Close'] > df['SMA_200']
            
            # RSI conditions
            ticker_data['oversold'] = df['RSI'] < 30
            ticker_data['overbought'] = df['RSI'] > 70
            
            # MACD crossovers
            ticker_data['macd_bullish'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
            ticker_data['macd_bearish'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
            
            # Volume trends
            ticker_data['volume_increasing'] = df['Volume'] > df['Volume_Prev']
            
            # Bollinger Bands
            ticker_data['above_bb_upper'] = df['Close'] > df['BB_Upper']
            ticker_data['below_bb_lower'] = df['Close'] < df['BB_Lower']
            ticker_data['bb_squeeze'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] < 0.1  # Less than 10% bandwidth
            
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
    market_breadth['pct_above_sma20'] = combined['above_sma20'].mean(axis=1) * 100
    market_breadth['pct_above_sma50'] = combined['above_sma50'].mean(axis=1) * 100
    market_breadth['pct_above_sma200'] = combined['above_sma200'].mean(axis=1) * 100
    
    # RSI breadth
    market_breadth['pct_oversold'] = combined['oversold'].mean(axis=1) * 100
    market_breadth['pct_overbought'] = combined['overbought'].mean(axis=1) * 100
    
    # MACD breadth
    market_breadth['pct_macd_bullish'] = combined['macd_bullish'].mean(axis=1) * 100
    market_breadth['pct_macd_bearish'] = combined['macd_bearish'].mean(axis=1) * 100
    
    # Volume breadth
    market_breadth['pct_volume_increasing'] = combined['volume_increasing'].mean(axis=1) * 100
    
    # Bollinger Bands breadth
    market_breadth['pct_above_bb_upper'] = combined['above_bb_upper'].mean(axis=1) * 100
    market_breadth['pct_below_bb_lower'] = combined['below_bb_lower'].mean(axis=1) * 100
    market_breadth['pct_bb_squeeze'] = combined['bb_squeeze'].mean(axis=1) * 100
    
    # Save to parquet
    output_path = Path(output_dir) / "daily_breadth.parquet"
    output_path.parent.mkdir(exist_ok=True)
    market_breadth.to_parquet(output_path)
    
    logger.info(f"Saved market breadth data to {output_path}")
    return market_breadth 
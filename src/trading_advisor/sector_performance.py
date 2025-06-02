"""Sector performance calculation."""

import logging
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
from tqdm import tqdm

from trading_advisor.data import load_tickers, normalize_ticker
from trading_advisor.features import load_features
from trading_advisor.market_features import load_sector_mapping

logger = logging.getLogger(__name__)

def calculate_sector_performance(
    tickers: Optional[List[str]] = None,
    features_dir: str = "features",
    market_features_dir: str = "market_features",
    output_dir: str = "market_features/sectors"
) -> pd.DataFrame:
    """
    Calculate sector performance metrics.
    
    Args:
        tickers: List of tickers to analyze. If None, uses all tickers in features_dir
        features_dir: Directory containing feature files
        market_features_dir: Directory containing market feature files (for sector mapping)
        output_dir: Directory to save sector performance data
        
    Returns:
        DataFrame with sector performance metrics, one row per date
    """
    # Load tickers if not provided
    if tickers is None:
        tickers = [f.stem.replace('_features', '') for f in Path(features_dir).glob('*_features.parquet')]
    logger.info(f"Processing {len(tickers)} tickers")
    
    # Load sector mapping
    sector_mapping = load_sector_mapping(market_features_dir)
    if sector_mapping.empty:
        logger.error("No sector mapping found")
        return pd.DataFrame()
    logger.info(f"Loaded sector mapping with {len(sector_mapping)} entries")
    
    # Initialize results DataFrame
    dates = None
    sector_data: Dict[str, List[pd.DataFrame]] = {}
    
    # Process each ticker
    for ticker in tqdm(tickers, desc="Calculating sector performance"):
        try:
            # Normalize ticker name
            norm_ticker = normalize_ticker(ticker)
            
            # Get sector info
            sector_info = sector_mapping[sector_mapping['ticker'] == norm_ticker]
            if sector_info.empty:
                logger.warning(f"No sector info for {ticker}")
                continue
                
            sector = sector_info.iloc[0]['sector']
            subsector = sector_info.iloc[0]['subsector']
            
            # Load features for this ticker
            df = load_features(norm_ticker, features_dir)
            if df.empty:
                logger.warning(f"No features found for {ticker}")
                continue
                
            # Get dates if not set
            if dates is None:
                dates = df.index
                
            # Calculate sector metrics for this ticker
            ticker_data = pd.DataFrame(index=dates)
            
            # Price metrics
            ticker_data['return_1d'] = df['Close'].pct_change()
            ticker_data['return_5d'] = df['Close'].pct_change(5)
            ticker_data['return_20d'] = df['Close'].pct_change(20)
            
            # Volume metrics
            ticker_data['volume_change'] = df['Volume'].pct_change()
            ticker_data['relative_volume'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # Technical indicators
            ticker_data['rsi'] = df['RSI']
            ticker_data['macd_hist'] = df['MACD_Hist']
            ticker_data['bb_position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Add to sector data
            if sector not in sector_data:
                sector_data[sector] = []
            sector_data[sector].append(ticker_data)
            
        except Exception as e:
            logger.warning(f"Failed to process {ticker}: {e}")
            continue
    
    if not sector_data:
        logger.error("No valid data found for any sectors")
        return pd.DataFrame()
    
    logger.info(f"Found data for {len(sector_data)} sectors")
    
    # Calculate sector-level metrics
    sector_performance_dict = {}
    
    # Process each sector
    for sector, ticker_dfs in sector_data.items():
        # Combine all ticker data for this sector
        sector_df = pd.concat(ticker_dfs, axis=1)
        
        # Calculate sector metrics
        sector_performance_dict[f'{sector}_return_1d'] = sector_df['return_1d'].mean(axis=1)
        sector_performance_dict[f'{sector}_return_5d'] = sector_df['return_5d'].mean(axis=1)
        sector_performance_dict[f'{sector}_return_20d'] = sector_df['return_20d'].mean(axis=1)
        sector_performance_dict[f'{sector}_volume_change'] = sector_df['volume_change'].mean(axis=1)
        sector_performance_dict[f'{sector}_relative_volume'] = sector_df['relative_volume'].mean(axis=1)
        sector_performance_dict[f'{sector}_avg_rsi'] = sector_df['rsi'].mean(axis=1)
        sector_performance_dict[f'{sector}_avg_macd_hist'] = sector_df['macd_hist'].mean(axis=1)
        sector_performance_dict[f'{sector}_avg_bb_position'] = sector_df['bb_position'].mean(axis=1)
        # Calculate sector momentum
        sector_performance_dict[f'{sector}_momentum'] = (
            sector_performance_dict[f'{sector}_return_1d'].rolling(5).mean() +
            sector_performance_dict[f'{sector}_return_5d'].rolling(5).mean() +
            sector_performance_dict[f'{sector}_return_20d'].rolling(5).mean()
        ) / 3
    
    # Calculate sector rotation metrics
    for sector in sector_data.keys():
        # Relative strength vs. market (using equal-weighted average of all sectors)
        market_return = pd.concat([sector_performance_dict[f'{s}_return_1d'] for s in sector_data.keys()], axis=1).mean(axis=1)
        sector_performance_dict[f'{sector}_relative_strength'] = (
            sector_performance_dict[f'{sector}_return_1d'] - market_return
        ).rolling(20).mean()
    
    # Concatenate all sector metrics into a single DataFrame
    sector_performance = pd.concat(sector_performance_dict, axis=1)
    
    # Save to parquet
    output_path = Path(output_dir) / "sector_performance.parquet"
    output_path.parent.mkdir(exist_ok=True)
    sector_performance.to_parquet(output_path)
    
    logger.info(f"Saved sector performance data to {output_path}")
    return sector_performance 
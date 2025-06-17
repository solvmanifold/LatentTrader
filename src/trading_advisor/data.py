"""Data handling and downloading functionality."""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import yfinance as yf
from functools import lru_cache
import os
import json
import re
import numpy as np

from trading_advisor.config import DATA_DIR, LOOKBACK_DAYS, REQUIRED_COLUMNS
from trading_advisor.analysis import calculate_technical_indicators

logger = logging.getLogger(__name__)

@lru_cache(maxsize=500)
def get_yf_ticker(ticker: str) -> yf.Ticker:
    """Get a cached Ticker object for the given symbol."""
    return yf.Ticker(ticker)

def ensure_data_dir():
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(exist_ok=True)

def get_current_date() -> datetime:
    """Get the current date."""
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

def read_csv_with_dates(file_path: Path, end_date: datetime) -> pd.DataFrame:
    """Read a CSV file and filter out future dates."""
    try:
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        df.index = pd.to_datetime(df.index).tz_localize(None)  # Remove timezone info
        return df[df.index.date <= end_date.date()]
    except Exception as e:
        logger.error(f"Error reading data from {file_path}: {e}")
        return pd.DataFrame()

def get_latest_data_date(ticker: str) -> Optional[datetime]:
    """Get the most recent date in the stored data for a ticker."""
    file_path = DATA_DIR / f"{ticker}.csv"
    if not file_path.exists():
        return None
    
    df = read_csv_with_dates(file_path, get_current_date())
    return df.index[-1] if len(df) > 0 else None

def is_csv_format_valid(file_path: Path, required_columns: list = None) -> bool:
    """Check if the CSV file has the expected columns."""
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS
    try:
        df = pd.read_csv(file_path, parse_dates=True, index_col=0, nrows=1)
        return all(col in df.columns for col in required_columns)
    except Exception as e:
        logger.error(f"Error validating CSV format for {file_path}: {e}")
        return False

def handle_multiindex_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Handle MultiIndex columns in the DataFrame."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
        
    df.columns = ['_'.join([str(i) for i in col if i]) for col in df.columns.values]
    ticker_cols = [col for col in df.columns if col.endswith(f'_{ticker}')]
    if ticker_cols:
        df = df[ticker_cols]
        df.columns = [col.replace(f'_{ticker}', '') for col in df.columns]
    return df

def normalize_ticker(ticker: str) -> str:
    """Normalize ticker format for Yahoo Finance."""
    # Convert to uppercase
    ticker = ticker.upper()
    # Replace dots with hyphens for class shares and other special cases
    if '.' in ticker:
        return ticker.replace('.', '-')
    return ticker

def get_features_path(ticker, features_dir="data/ticker_features"):
    return Path(features_dir) / f"{ticker}_features.parquet"

def fill_missing_trading_days(df: pd.DataFrame, reference_df: pd.DataFrame, data_type: str = 'price') -> pd.DataFrame:
    """Fill in missing trading days in the DataFrame based on the reference DataFrame.
    
    Args:
        df: DataFrame to fill missing trading days in.
        reference_df: Reference DataFrame with the set of trading days.
        data_type: Type of data being filled ('price', 'returns', 'volume', or 'sentiment').
        
    Returns:
        DataFrame with missing trading days filled in according to data type rules.
    """
    if df.empty or reference_df.empty:
        return df
    
    # Extract the set of trading days from the reference DataFrame
    trading_days = reference_df.index.unique()
    
    # Identify missing trading days in the target DataFrame
    missing_days = trading_days.difference(df.index)
    
    if missing_days.empty:
        return df
    
    # Create a DataFrame for missing days with NaN values
    missing_df = pd.DataFrame(index=missing_days, columns=df.columns)
    
    # Combine the original DataFrame with the missing days DataFrame
    filled_df = pd.concat([df, missing_df])
    filled_df = filled_df.sort_index()
    
    # Apply appropriate forward filling based on data type
    if data_type == 'price':
        # Forward fill price data
        filled_df = filled_df.ffill()
    elif data_type == 'returns':
        # Forward fill returns data with limit=1 to prevent artificial smoothing
        filled_df = filled_df.ffill(limit=1)
    elif data_type == 'sentiment':
        # Forward fill sentiment data with limit=5 to maintain recent context
        filled_df = filled_df.ffill(limit=5)
    # For volume data, we don't forward fill (keep NaN values)
    
    return filled_df

def download_stock_data(
    ticker: str,
    history_days: int = LOOKBACK_DAYS,
    max_retries: int = 3,
    features_dir: str = "data/ticker_features",
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    features_filename: str = None  # New argument for test isolation
) -> pd.DataFrame:
    """Parquet-first: Check for up-to-date Parquet, else download/append missing data and update features."""
    features_dir = Path(features_dir)
    features_dir.mkdir(exist_ok=True)
    if features_filename:
        features_path = features_dir / features_filename
    else:
        features_path = features_dir / f"{ticker}_features.parquet"
    
    # Try to load existing data first
    df = pd.DataFrame()
    if features_path.exists():
        try:
            df = pd.read_parquet(features_path)
            if not df.empty:
                df.index = pd.to_datetime(df.index).normalize()
        except Exception as e:
            logger.warning(f"Error reading {features_path}: {e}")
    
    # Calculate date range for new data
    if end_date is not None:
        fetch_end = pd.to_datetime(end_date).normalize()
    else:
        fetch_end = pd.Timestamp.today().normalize()
    if not df.empty:
        last_date = df.index.max().normalize()
        fetch_start = last_date + pd.Timedelta(days=1)
    else:
        if start_date is not None:
            fetch_start = pd.to_datetime(start_date).normalize()
        else:
            fetch_start = fetch_end - pd.Timedelta(days=history_days)
    
    # If we have data and it's up to date, return it
    if not df.empty and fetch_start > fetch_end:
        logger.info(f"No new data to download for {ticker}. Data is up to date through {df.index.max().date()}.")
        return df
    
    # Download new data with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to download data for {ticker} (attempt {attempt+1}) from {fetch_start.date()} to {fetch_end.date()}")
            ticker_obj = get_yf_ticker(ticker)
            new_df = ticker_obj.history(
                start=fetch_start,
                end=fetch_end + pd.Timedelta(days=1),
                auto_adjust=True
            )
            if len(new_df) > 0:
                new_df.index = pd.to_datetime(new_df.index).tz_localize(None).normalize()
                # Standardize column names to lowercase immediately after download
                new_df.columns = [c.lower() for c in new_df.columns]
                
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in new_df.columns]
                if missing_cols:
                    logger.error(f"Missing required columns for {ticker}: {missing_cols}")
                    raise ValueError(f"Missing required columns: {missing_cols}")
                
                # Handle adj_close column (optional)
                if 'adj close' in new_df.columns:
                    new_df['adj_close'] = new_df['adj close']
                    new_df = new_df.drop(columns=['adj close'])
                elif 'adj_close' not in new_df.columns:
                    # If adj_close is missing, use close price
                    new_df['adj_close'] = new_df['close']
                
                # Only keep truly new rows
                if not df.empty:
                    new_rows = new_df[~new_df.index.isin(df.index)]
                else:
                    new_rows = new_df
                if new_rows.empty:
                    logger.info(f"No new rows to append for {ticker}. Data is already up to date.")
                    return df
                
                # Merge with existing data
                merged_df = pd.concat([df, new_rows])
                # Set columns to canonical list from new_rows, then deduplicate
                merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
                merged_df = merged_df[new_rows.columns.tolist()]
                merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
                merged_df = merged_df.sort_index()
                logger.info(f"Appended {len(new_rows)} new rows for {ticker} (now {len(merged_df)} total rows).")
                
                # Handle missing trading days for different data types
                # Price data: Forward fill
                price_cols = ['open', 'high', 'low', 'close', 'adj_close']
                price_df = merged_df[price_cols].copy()
                price_df = fill_missing_trading_days(price_df, merged_df, data_type='price')
                
                # Volume data: Keep NaN values
                volume_cols = ['volume']
                volume_df = merged_df[volume_cols].copy()
                volume_df = fill_missing_trading_days(volume_df, merged_df, data_type='volume')
                
                # Returns data: Forward fill with limit=1
                returns_cols = [col for col in merged_df.columns if 'returns' in col]
                if returns_cols:
                    returns_df = merged_df[returns_cols].copy()
                    returns_df = fill_missing_trading_days(returns_df, merged_df, data_type='returns')
                
                # Drop all technical indicator columns before recalculating
                indicator_cols = [
                    'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'bb_upper', 'bb_lower', 'bb_middle', 'bb_pband',
                    'sma_20', 'sma_50', 'sma_100', 'sma_200',
                    'ema_100', 'ema_200'
                ]
                merged_df = merged_df.drop(columns=[col for col in indicator_cols if col in merged_df.columns], errors='ignore')
                
                # Recalculate technical indicators
                merged_df = calculate_technical_indicators(merged_df)
                
                # Final deduplication to be safe
                merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
                
                # Standardize columns and date before saving
                merged_df = standardize_columns_and_date(merged_df)
                
                # Add volume_prev using shift(1) without ffill to preserve gaps
                merged_df['volume_prev'] = merged_df['volume'].shift(1)
                # Reorder columns to place volume_prev right after volume
                cols = merged_df.columns.tolist()
                volume_idx = cols.index('volume')
                cols.remove('volume_prev')
                cols.insert(volume_idx + 1, 'volume_prev')
                merged_df = merged_df[cols]
                
                merged_df.to_parquet(features_path)
                return merged_df
            else:
                logger.info(f"No data returned for {ticker} from yfinance.")
            import time; time.sleep(2)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to download {ticker} after {max_retries} attempts: {e}")
            import time; time.sleep(2 ** (attempt + 1))
    logger.warning(f"All download attempts failed for {ticker}. Returning existing data if available.")
    return df

def parse_brokerage_csv(file_path: Path) -> Dict[str, Dict]:
    """Parse the brokerage CSV file and return a dictionary of positions."""
    try:
        # Skip the first two rows (header and empty row)
        df = pd.read_csv(file_path, skiprows=2)
        
        # Filter out non-equity positions (Cash, Account Total, etc.)
        equity_positions = df[df['Security Type'] == 'Equity']
        
        positions = {}
        for _, row in equity_positions.iterrows():
            symbol = row['Symbol']
            # Clean up numeric values by removing $ and % symbols and commas
            price = float(str(row['Price']).replace('$', '').replace(',', ''))
            market_value = float(str(row['Mkt Val (Market Value)']).replace('$', '').replace(',', ''))
            cost_basis = float(str(row['Cost Basis']).replace('$', '').replace(',', ''))
            gain_pct = float(str(row['Gain % (Gain/Loss %)']).replace('%', ''))
            account_pct = float(str(row['% of Acct (% of Account)']).replace('%', ''))
            
            positions[symbol] = {
                'quantity': float(row['Qty (Quantity)']),
                'price': price,
                'market_value': market_value,
                'cost_basis': cost_basis,
                'gain_pct': gain_pct,
                'account_pct': account_pct
            }
        
        return positions
    except Exception as e:
        logger.error(f"Error parsing brokerage CSV {file_path}: {e}")
        return {}

def load_positions(positions_file: Optional[Path]) -> Dict[str, Dict]:
    """Load current positions from brokerage CSV file."""
    if not positions_file or not positions_file.exists():
        return {}
    
    return parse_brokerage_csv(positions_file)

def load_tickers(tickers_input: Optional[Path]) -> list[str]:
    """Load tickers from input file or use default."""
    if tickers_input is None:
        # Default to "all" to get S&P 500 tickers
        return load_tickers("all")
    elif str(tickers_input) == "all":
        # Fetch all S&P 500 tickers from Wikipedia
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Fetch the S&P 500 components from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the first table (which contains the S&P 500 components)
            table = soup.find('table', {'class': 'wikitable'})
            if not table:
                logger.error("Could not find S&P 500 components table on Wikipedia")
                sys.exit(1)
            
            # Extract tickers from the table
            tickers = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if cells:
                    ticker = cells[0].text.strip()
                    tickers.append(ticker)
            
            if not tickers:
                logger.error("No tickers found in S&P 500 components table")
                sys.exit(1)
            
            return tickers
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            sys.exit(1)
    else:
        # Load tickers from the provided text file
        try:
            with open(tickers_input, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            return tickers
        except Exception as e:
            logger.error(f"Error reading tickers from {tickers_input}: {e}")
            sys.exit(1)

def load_ticker_features(tickers: List[str], features_dir: str) -> pd.DataFrame:
    """Load and combine feature data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        features_dir: Directory containing feature Parquet files
        
    Returns:
        DataFrame containing features for all tickers
    """
    dfs = []
    for ticker in tickers:
        features_path = Path(features_dir) / f"{ticker}_features.parquet"
        if not features_path.exists():
            logger.warning(f"Features file not found for {ticker}: {features_path}")
            continue
        try:
            df = pd.read_parquet(features_path)
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
                df.index = pd.to_datetime(df.index)
            df['ticker'] = ticker  # Add ticker column
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading features for {ticker}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=False)
    
    return combined_df 

def standardize_columns_and_date(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame columns and date handling.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized columns and date index
    """
    # Handle date column first
    if 'date' in df.columns:
        df = df.set_index('date')
    elif 'Date' in df.columns:
        df = df.set_index('Date')
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have either a DatetimeIndex or a 'date'/'Date' column")
    
    # Drop any other date columns
    date_cols = [col for col in df.columns if col.lower() in ['date', 'dates', 'datetime', 'timestamp']]
    if date_cols:
        df = df.drop(columns=date_cols)
    
    # Standardize column names
    # First convert to lowercase and replace spaces with underscores
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    # Then replace any remaining special characters with underscores
    df.columns = [''.join(c if c.isalnum() else '_' for c in col) for col in df.columns]
    # Remove consecutive underscores
    df.columns = [col.replace('__', '_') for col in df.columns]
    # Remove leading/trailing underscores
    df.columns = [col.strip('_') for col in df.columns]
    
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Normalize dates (set time to midnight)
    df.index = df.index.normalize()
    
    return df

def calculate_market_features(
    tickers: List[str],
    days: int = LOOKBACK_DAYS,
    features_dir: str = "data/market_features",
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None
) -> pd.DataFrame:
    """Calculate market-wide features from ticker data."""
    features_dir = Path(features_dir)
    features_dir.mkdir(exist_ok=True)
    features_path = features_dir / "market_features.parquet"
    
    # Try to load existing data first
    df = pd.DataFrame()
    if features_path.exists():
        try:
            df = pd.read_parquet(features_path)
            if not df.empty:
                df.index = pd.to_datetime(df.index).normalize()
        except Exception as e:
            logger.warning(f"Error reading {features_path}: {e}")
    
    # Calculate date range for new data
    if end_date is not None:
        fetch_end = pd.to_datetime(end_date).normalize()
    else:
        fetch_end = pd.Timestamp.today().normalize()
    if not df.empty:
        last_date = df.index.max().normalize()
        fetch_start = last_date + pd.Timedelta(days=1)
    else:
        if start_date is not None:
            fetch_start = pd.to_datetime(start_date).normalize()
        else:
            fetch_start = fetch_end - pd.Timedelta(days=days)
    
    # If we have data and it's up to date, return it
    if not df.empty and fetch_start > fetch_end:
        logger.info(f"No new market data to calculate. Data is up to date through {df.index.max().date()}.")
        return df
    
    # Download data for all tickers
    ticker_data = {}
    for ticker in tickers:
        ticker_df = download_stock_data(
            ticker,
            history_days=days,
            features_dir="data/ticker_features",
            start_date=fetch_start,
            end_date=fetch_end
        )
        if not ticker_df.empty:
            ticker_data[ticker] = ticker_df
    
    if not ticker_data:
        logger.warning("No ticker data available for market feature calculation.")
        return df
    
    # Calculate market features
    market_features = {}
    
    # Market breadth indicators
    market_features['advancing_stocks'] = calculate_advancing_stocks(ticker_data)
    market_features['declining_stocks'] = calculate_declining_stocks(ticker_data)
    market_features['adv_dec_ratio'] = calculate_adv_dec_ratio(ticker_data)
    market_features['new_highs'] = calculate_new_highs(ticker_data)
    market_features['new_lows'] = calculate_new_lows(ticker_data)
    market_features['high_low_ratio'] = calculate_high_low_ratio(ticker_data)
    
    # S&P 500 data
    sp500_data = download_stock_data(
        '^GSPC',
        history_days=days,
        features_dir="data/ticker_features",
        start_date=fetch_start,
        end_date=fetch_end
    )
    if not sp500_data.empty:
        market_features['sp500_close'] = sp500_data['close']
        market_features['sp500_volume'] = sp500_data['volume']
        market_features['sp500_returns'] = sp500_data['returns']
    
    # Sector performance
    sector_performance = calculate_sector_performance(ticker_data)
    market_features.update(sector_performance)
    
    # Market volatility
    market_features['market_volatility'] = calculate_market_volatility(ticker_data)
    
    # Market sentiment
    market_features['market_sentiment'] = calculate_market_sentiment(ticker_data)
    
    # Create DataFrame from market features
    new_df = pd.DataFrame(market_features)
    new_df.index = pd.to_datetime(new_df.index).normalize()
    
    # Handle missing trading days for different data types
    # Price data: Forward fill
    price_cols = ['sp500_close']
    price_df = new_df[price_cols].copy()
    price_df = fill_missing_trading_days(price_df, new_df, data_type='price')
    
    # Volume data: Keep NaN values
    volume_cols = ['sp500_volume']
    volume_df = new_df[volume_cols].copy()
    volume_df = fill_missing_trading_days(volume_df, new_df, data_type='volume')
    
    # Returns data: Forward fill with limit=1
    returns_cols = ['sp500_returns']
    returns_df = new_df[returns_cols].copy()
    returns_df = fill_missing_trading_days(returns_df, new_df, data_type='returns')
    
    # Sentiment data: Forward fill with limit=5
    sentiment_cols = ['market_sentiment']
    sentiment_df = new_df[sentiment_cols].copy()
    sentiment_df = fill_missing_trading_days(sentiment_df, new_df, data_type='sentiment')
    
    # Market breadth indicators: Keep NaN values
    breadth_cols = [
        'advancing_stocks', 'declining_stocks', 'adv_dec_ratio',
        'new_highs', 'new_lows', 'high_low_ratio'
    ]
    breadth_df = new_df[breadth_cols].copy()
    breadth_df = fill_missing_trading_days(breadth_df, new_df, data_type='volume')
    
    # Sector performance: Keep NaN values
    sector_cols = [col for col in new_df.columns if col.startswith('sector_')]
    sector_df = new_df[sector_cols].copy()
    sector_df = fill_missing_trading_days(sector_df, new_df, data_type='volume')
    
    # Market volatility: Keep NaN values
    volatility_cols = ['market_volatility']
    volatility_df = new_df[volatility_cols].copy()
    volatility_df = fill_missing_trading_days(volatility_df, new_df, data_type='volume')
    
    # Merge all dataframes
    new_df = pd.concat([
        price_df, volume_df, returns_df, sentiment_df,
        breadth_df, sector_df, volatility_df
    ], axis=1)
    
    # Merge with existing data
    if not df.empty:
        merged_df = pd.concat([df, new_df])
        merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
        merged_df = merged_df.sort_index()
    else:
        merged_df = new_df
    
    # Save to parquet
    merged_df.to_parquet(features_path)
    
    return merged_df 
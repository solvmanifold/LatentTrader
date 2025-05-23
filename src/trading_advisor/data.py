"""Data handling and downloading functionality."""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf
from functools import lru_cache
import os
import json

from trading_advisor.config import DATA_DIR, LOOKBACK_DAYS, REQUIRED_COLUMNS
from trading_advisor.analysis import calculate_technical_indicators, get_analyst_targets

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
    # Replace dots with hyphens for class B shares
    if ticker.endswith('.B'):
        return ticker.replace('.B', '-B')
    return ticker

def get_features_path(ticker, features_dir="features"):
    return Path(features_dir) / f"{ticker}_features.parquet"

def download_stock_data(
    ticker: str,
    history_days: int = LOOKBACK_DAYS,
    max_retries: int = 3,
    features_dir: str = "features",
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
                merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
                merged_df = merged_df.sort_index()
                logger.info(f"Appended {len(new_rows)} new rows for {ticker} (now {len(merged_df)} total rows).")
                # Recalculate technical indicators for all (or just new rows if you want to optimize)
                merged_df = calculate_technical_indicators(merged_df)
                # Get analyst targets before saving
                analyst_targets = get_analyst_targets(ticker)
                # --- Analyst targets propagation fix ---
                if 'analyst_targets' not in merged_df.columns:
                    merged_df['analyst_targets'] = None
                # Only set None for new rows, preserve existing values
                if not df.empty and 'analyst_targets' in df.columns:
                    # Copy over old values for existing rows
                    for idx in df.index:
                        if idx in merged_df.index:
                            merged_df.at[idx, 'analyst_targets'] = df.at[idx, 'analyst_targets']
                # Set None for new rows except last, set analyst_targets for last row
                for idx in new_rows.index:
                    merged_df.at[idx, 'analyst_targets'] = None
                if analyst_targets and not merged_df.empty:
                    merged_df.at[merged_df.index[-1], 'analyst_targets'] = json.dumps(analyst_targets)
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
        return ["AAPL"]  # Default to AAPL if no tickers provided
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
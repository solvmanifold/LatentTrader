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

from config import DATA_DIR, LOOKBACK_DAYS, REQUIRED_COLUMNS

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

def download_stock_data(ticker: str, history_days: int = LOOKBACK_DAYS, max_retries: int = 3) -> pd.DataFrame:
    """Download stock data for a ticker, only getting new data if available."""
    # Normalize ticker format
    normalized_ticker = normalize_ticker(ticker)
    file_path = DATA_DIR / f"{ticker}.csv"  # Keep original ticker for filename
    end_date = get_current_date()
    
    # If we have existing data, read it first
    if file_path.exists():
        if not is_csv_format_valid(file_path):
            logger.error(f"The file {file_path} is not in the expected format. Please delete or fix it before running the script.")
            sys.exit(1)
            
        try:
            # Read existing data and filter out future dates
            df = read_csv_with_dates(file_path, end_date)
            if not df.empty:
                latest_date = df.index[-1].date()
                # If we already have data up to yesterday, just return it
                if latest_date >= end_date.date():
                    return df
                
                # Otherwise, start from the day after our latest data
                start_date = datetime.combine(latest_date, datetime.min.time())
            else:
                # If no valid data, start from history_days ago
                start_date = end_date - timedelta(days=history_days)
        except Exception as e:
            logger.error(f"Error reading existing data for {ticker}: {e}")
            # If there's an error reading existing data, start fresh
            start_date = end_date - timedelta(days=history_days)
    else:
        # If no existing data, start from history_days ago
        start_date = end_date - timedelta(days=history_days)
    
    # Ensure we're not trying to download future data
    if start_date.date() > end_date.date():
        return read_csv_with_dates(file_path, end_date) if file_path.exists() else pd.DataFrame()
    
    for attempt in range(max_retries):
        try:
            # Download new data using cached Ticker object
            ticker_obj = get_yf_ticker(normalized_ticker)
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                auto_adjust=True
            )
            
            if len(df) > 0:  # Check if DataFrame has any rows
                # Ensure index is datetime and remove timezone info
                df.index = pd.to_datetime(df.index).tz_localize(None)
                
                # Filter out any future dates
                df = df[df.index.date <= end_date.date()]
                
                if file_path.exists():
                    try:
                        # Read existing data
                        existing_df = read_csv_with_dates(file_path, end_date)
                        
                        # Combine data
                        df = pd.concat([existing_df, df])
                        df = df[~df.index.duplicated(keep='last')]
                        df = df.sort_index()
                    except Exception as e:
                        logger.error(f"Error processing existing data for {ticker}: {e}")
                        # If there's an error with existing data, just use the new data
                        pass
                
                try:
                    # Save to CSV with explicit date format
                    df.to_csv(file_path, date_format='%Y-%m-%d')
                    return df
                except Exception as e:
                    logger.error(f"Error saving data for {ticker}: {e}")
                    return df  # Return the DataFrame even if saving fails
                
            time.sleep(2)  # Increased delay between attempts
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to download {ticker} after {max_retries} attempts: {e}")
            time.sleep(2 ** (attempt + 1))  # Exponential backoff with longer initial delay
    
    # If all attempts failed, try to return existing data
    return read_csv_with_dates(file_path, end_date) if file_path.exists() else pd.DataFrame()

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

def load_tickers(tickers_input: Optional[str]) -> list[str]:
    """Load tickers from input file or use default."""
    if tickers_input is None:
        return ["AAPL"]  # Default to AAPL if no tickers provided
    elif tickers_input == "all":
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
                tickers = [line.strip() for line in f if line.strip()]  # Skip empty lines
            if not tickers:
                logger.error(f"No tickers found in {tickers_input}")
                sys.exit(1)
            return tickers
        except Exception as e:
            logger.error(f"Error loading tickers from {tickers_input}: {e}")
            sys.exit(1) 
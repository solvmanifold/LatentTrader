#!/usr/bin/env python3

import os
import typer
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import time
import sys
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = typer.Typer()

# Constants
DATA_DIR = Path("data")
LOOKBACK_DAYS = 100
REQUIRED_COLUMNS = ["Close", "High", "Low", "Open", "Volume"]

@lru_cache(maxsize=500)
def get_yf_ticker(ticker: str) -> yf.Ticker:
    """Get a cached Ticker object for the given symbol."""
    return yf.Ticker(ticker)

def ensure_data_dir():
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(exist_ok=True)

def get_current_date() -> datetime:
    """Get the current date"""
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

def is_csv_format_valid(file_path: Path, required_columns: List[str] = None) -> bool:
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

def download_stock_data(ticker: str, history_days: int = LOOKBACK_DAYS, max_retries: int = 3) -> pd.DataFrame:
    """Download stock data for a ticker, only getting new data if available."""
    file_path = DATA_DIR / f"{ticker}.csv"
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
            ticker_obj = get_yf_ticker(ticker)
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
                
            time.sleep(1)  # Add a small delay between attempts
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to download {ticker} after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    # If all attempts failed, try to return existing data
    return read_csv_with_dates(file_path, end_date) if file_path.exists() else pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the stock data."""
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    
    # RSI
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Volume Analysis
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    return df

def get_analyst_targets(ticker: str) -> Optional[Dict]:
    """Get analyst price targets for a ticker."""
    try:
        ticker_obj = get_yf_ticker(ticker)
        targets = ticker_obj.analyst_price_targets
        
        if targets is None or not isinstance(targets, dict):
            return None
            
        # Get current price from the targets dict
        current_price = targets.get('current')
        if current_price is None:
            return None
            
        # Validate all required fields are present and not None
        required_fields = ['median', 'low', 'high', 'current']
        if not all(field in targets and targets[field] is not None for field in required_fields):
            return None
            
        return {
            'median': targets['median'],
            'mean': targets.get('mean'),  # Optional field
            'low': targets['low'],
            'high': targets['high'],
            'current_price': targets['current']
        }
    except Exception as e:
        logger.error(f"Error getting analyst targets for {ticker}: {e}")
        return None

def calculate_score(df: pd.DataFrame, ticker: str) -> float:
    """Calculate a technical score based on indicators."""
    score = 0
    latest = df.iloc[-1]
    
    # RSI scoring
    if latest['RSI'] < 30:
        score += 2
    elif latest['RSI'] > 70:
        score += 1
    
    # Bollinger Bands scoring
    if latest['Close'] > latest['BB_upper']:
        score += 2
    elif latest['Close'] < latest['BB_lower']:
        score += 2
    
    # MACD analysis
    if len(df) >= 5:  # Ensure we have enough data for trend analysis
        prev = df.iloc[-2]
        # Check for crossover
        if prev['MACD'] < prev['MACD_signal'] and latest['MACD'] > latest['MACD_signal']:
            score += 2
        elif prev['MACD'] > prev['MACD_signal'] and latest['MACD'] < latest['MACD_signal']:
            score += 2
        
        # Check for sustained trend
        macd_trend = df['MACD'][-5:].mean() - df['MACD_signal'][-5:].mean()
        if abs(macd_trend) > 1.0:  # Strong divergence
            score += 2
        elif abs(macd_trend) > 0.5:  # Moderate divergence
            score += 1
        
        # Check for trend acceleration
        if len(df) >= 10:
            recent_trend = df['MACD'][-5:].mean() - df['MACD_signal'][-5:].mean()
            prev_trend = df['MACD'][-10:-5].mean() - df['MACD_signal'][-10:-5].mean()
            if abs(recent_trend) > abs(prev_trend) * 1.5:  # Trend is accelerating
                score += 1
    
    # Volume spike
    if latest['Volume_Ratio'] > 2:
        score += 1
    
    # Analyst target scoring
    targets = get_analyst_targets(ticker)
    if targets and all(k in targets and targets[k] is not None for k in ['median', 'current_price']):
        current_price = targets['current_price']
        median_target = targets['median']
        
        # Calculate upside percentage
        upside_pct = ((median_target - current_price) / current_price) * 100
        
        # Score based on upside potential
        if upside_pct >= 20:
            score += 2
        elif upside_pct >= 10:
            score += 1
    
    return score

def parse_brokerage_csv(file_path: Path) -> Dict[str, Dict]:
    """Parse the brokerage CSV file and return a dictionary of positions."""
    try:
        # Skip the first two rows (header and empty row)
        df = pd.read_csv(file_path, skiprows=2)
        
        # Filter out non-equity positions (Cash, Account Total, etc.)
        equity_positions = df[df['Security Type'].isin(['Equity', 'ETFs & Closed End Funds'])]
        
        positions = {}
        for _, row in equity_positions.iterrows():
            symbol = row['Symbol']
            positions[symbol] = {
                'quantity': float(row['Qty (Quantity)']),
                'price': float(row['Price'].replace('$', '').replace(',', '')),
                'market_value': float(row['Mkt Val (Market Value)'].replace('$', '').replace(',', '')),
                'cost_basis': float(row['Cost Basis'].replace('$', '').replace(',', '')),
                'gain_pct': float(row['Gain % (Gain/Loss %)'].replace('%', '')),
                'account_pct': float(row['% of Acct (% of Account)'].replace('%', ''))
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

def generate_technical_summary(ticker: str, df: pd.DataFrame) -> str:
    """Generate a technical analysis summary for a stock using markdown formatting."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    # Get recent price movement
    price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
    price_trend = "up" if price_change > 0 else "down"
    
    # Build the summary with markdown formatting
    summary = f"**${ticker}**\n"
    
    # Add analyst targets if available
    targets = get_analyst_targets(ticker)
    if targets and all(k in targets and targets[k] is not None for k in ['median', 'low', 'high', 'current_price']):
        current_price = targets['current_price']
        median_target = targets['median']
        low_target = targets['low']
        high_target = targets['high']
        
        # Calculate upside percentage
        upside_pct = ((median_target - current_price) / current_price) * 100
        
        summary += f"- Current price: ${current_price:.2f}\n"
        summary += f"- Analyst target: ${median_target:.2f} median (range: ${low_target:.2f}–${high_target:.2f})\n"
        summary += f"- Implied upside: {upside_pct:+.1f}%\n"
    
    # RSI analysis
    rsi_status = "oversold" if latest['RSI'] < 30 else "overbought" if latest['RSI'] > 70 else "neutral"
    summary += f"- RSI: {latest['RSI']:.1f} ({rsi_status})\n"
    
    # MACD analysis
    macd_diff = latest['MACD'] - latest['MACD_signal']
    if abs(macd_diff) < 0.1:
        macd_status = "neutral"
    else:
        macd_status = "bullish" if macd_diff > 0 else "bearish"
    
    # Add MACD trend context
    if len(df) >= 5:
        macd_trend = df['MACD'][-5:].mean() - df['MACD_signal'][-5:].mean()
        trend_context = []
        
        # Add trend strength
        if abs(macd_trend) > 1.0:
            trend_context.append("strong")
        elif abs(macd_trend) > 0.5:
            trend_context.append("moderate")
            
        # Add trend direction
        if macd_trend > 0:
            trend_context.append("divergence")
        else:
            trend_context.append("convergence")
            
        # Add trend acceleration if available
        if len(df) >= 10:
            recent_trend = df['MACD'][-5:].mean() - df['MACD_signal'][-5:].mean()
            prev_trend = df['MACD'][-10:-5].mean() - df['MACD_signal'][-10:-5].mean()
            if abs(recent_trend) > abs(prev_trend) * 1.5:
                trend_context.append("accelerating")
            elif abs(recent_trend) < abs(prev_trend) * 0.5:
                trend_context.append("decelerating")
        
        if trend_context:
            macd_status += f" with {' '.join(trend_context)}"
    
    summary += f"- MACD: {macd_status}\n"
    
    # Volume analysis
    if latest['Volume_Ratio'] > 2:
        summary += "- Volume: high\n"
    elif latest['Volume_Ratio'] < 0.5:
        summary += "- Volume: low\n"
    
    # Bollinger Bands analysis
    if latest['Close'] > latest['BB_upper']:
        summary += "- Price: above upper BB\n"
    elif latest['Close'] < latest['BB_lower']:
        summary += "- Price: below lower BB\n"
    
    # Moving averages
    ma_trend = "bullish" if latest['MA20'] > latest['MA50'] else "bearish"
    summary += f"- MA trend: {ma_trend}\n"
    
    # Recent price movement
    summary += f"- 5d change: {price_change:+.1f}%\n"
    
    return summary

def generate_position_summary(position_info: Dict) -> str:
    """Generate a summary of the current position using markdown formatting."""
    summary = f"- **Position**: {position_info['quantity']} shares @ ${position_info['cost_basis']:.2f}, "
    summary += f"{position_info['gain_pct']:+.1f}%, "
    summary += f"{position_info['account_pct']:.1f}% of account"
    return summary

def generate_summary(ticker: str, df: pd.DataFrame, position_info: Optional[Dict] = None) -> str:
    """Generate a complete summary for a stock, including technical analysis and position info if available."""
    summary = generate_technical_summary(ticker, df)
    if position_info:
        summary += generate_position_summary(position_info)
    return summary

def load_tickers(tickers_input: Optional[str]) -> List[str]:
    """Load tickers from input file or use default."""
    if tickers_input is None:
        return ["AAPL"]  # Default to AAPL if no tickers provided
    elif tickers_input == "all":
        # Load all S&P 500 tickers from a predefined list or CSV
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM", "V", "JNJ"]  # Example list
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

def process_ticker(ticker: str, current_positions: Dict[str, Dict], history_days: int = LOOKBACK_DAYS) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Process a single ticker and return its summary and score."""
    try:
        df = download_stock_data(ticker, history_days)
        if df is None or df.empty or not all(col in df.columns for col in REQUIRED_COLUMNS):
            logger.warning(f"Skipping {ticker}: No data or missing columns.")
            return None, None, None
        
        df = calculate_indicators(df)
        score = calculate_score(df, ticker)
        
        if ticker in current_positions:
            return ticker, score, generate_summary(ticker, df, current_positions[ticker])
        else:
            return ticker, score, generate_summary(ticker, df)
            
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return None, None, None

@app.command()
def main(
    top_n: int = typer.Option(5, help="Number of new tickers to include"),
    output: Optional[Path] = typer.Option(None, help="Path to write the markdown file"),
    positions: Optional[Path] = typer.Option(None, help="Path to brokerage positions CSV"),
    tickers: Optional[str] = typer.Option(None, help="Path to tickers.txt file (one ticker per line), or 'all' for all S&P 500 tickers"),
    history_days: int = typer.Option(100, help="Days of historical data to fetch"),
    include_charts: bool = typer.Option(False, help="Include plots in the summary output"),
    positions_only: bool = typer.Option(False, help="Only analyze current positions, ignore other tickers"),
    no_positions: bool = typer.Option(False, help="Exclude current positions from analysis")
):
    """Generate a weekly trading advisor report."""
    ensure_data_dir()
    
    # Load current positions
    current_positions = load_positions(positions)
    
    # Load tickers
    tickers_list = load_tickers(tickers)
    
    # Handle position-only and no-positions flags
    if positions_only:
        if not current_positions:
            logger.error("No positions found. Please provide a positions file with --positions.")
            sys.exit(1)
        tickers_list = list(current_positions.keys())
    elif no_positions:
        tickers_list = [t for t in tickers_list if t not in current_positions]
    else:
        # Add any current positions that aren't in the tickers list
        for ticker in current_positions:
            if ticker not in tickers_list:
                tickers_list.append(ticker)
    
    # Process all stocks
    stock_scores = []
    position_summaries = []
    
    with typer.progressbar(tickers_list, label="Processing tickers") as progress:
        for ticker in progress:
            ticker, score, summary = process_ticker(ticker, current_positions, history_days)
            if ticker and score is not None and summary:
                if ticker in current_positions:
                    position_summaries.append(summary)
                else:
                    stock_scores.append((ticker, score, summary))
    
    # Sort and select top N stocks
    stock_scores.sort(key=lambda x: x[1], reverse=True)
    top_stocks = stock_scores[:top_n]
    
    # Generate the report
    if positions_only:
        report = "You are an equity analyst. Here is the technical summary for my current holdings.\n\n"
        report += "Please analyze each position and advise whether to hold, sell, or adjust (e.g. trailing stop). "
        report += "Consider both technical indicators and analyst targets in your recommendations.\n\n"
    elif no_positions:
        report = "You are an equity analyst. Here are new technical picks from the S&P 500.\n\n"
        report += "Please assess each stock for trade viability, considering both technical indicators and analyst targets. "
        report += "For each pick, suggest an appropriate entry strategy and risk management approach.\n\n"
    else:
        report = "You are an equity analyst. Here is the technical summary for several S&P 500 stocks.\n\n"
        report += "The first section contains current positions. Please advise whether to hold, sell, or adjust (e.g. trailing stop).\n"
        report += "The second section contains new picks flagged by our model this week. Please assess each for trade viability.\n\n"
    
    if not no_positions and position_summaries:
        report += "---\n\n"
        report += "### 📊 Current Positions (Hold/Sell Guidance)\n\n"
        report += "---\n\n"
        report += "\n\n".join(position_summaries) + "\n\n"
    
    if not positions_only and top_stocks:
        report += "---\n\n"
        report += "### 🚀 New Technical Picks (Trade Candidates)\n\n"
        report += "---\n\n"
        for ticker, score, summary in top_stocks:
            report += summary + "\n"
    
    # TODO: Add chart generation when include_charts is True
    if include_charts:
        logger.info("Chart generation is not yet implemented")
    
    # Output the report
    if output:
        output.write_text(report)
    else:
        print(report)

if __name__ == "__main__":
    app() 
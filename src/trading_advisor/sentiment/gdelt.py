"""
GDELT news sentiment data collection module.

This module handles the collection and processing of news sentiment data from GDELT.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
import requests
from datetime import datetime, timedelta
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

class GDELTClient:
    """Client for fetching GDELT sentiment data."""
    
    def __init__(self, data_dir: Path):
        """Initialize GDELT client.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        
    def fetch_gdelt_data(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch GDELT sentiment data for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD or YYYYMMDD format
            end_date: Optional end date in YYYY-MM-DD or YYYYMMDD format. If None, uses start_date
            
        Returns:
            DataFrame with daily sentiment data
        """
        if end_date is None:
            end_date = start_date
            
        # Convert dates to datetime, handling both formats
        def parse_date(date_str: str) -> datetime:
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return datetime.strptime(date_str, "%Y%m%d")
            
        start = parse_date(start_date)
        end = parse_date(end_date)
        
        # Generate date range
        date_range = pd.date_range(start=start, end=end, freq='D')
        sentiment_data = []
        
        base_url = "http://data.gdeltproject.org/events/"
        
        for single_date in tqdm(date_range, desc="Fetching GDELT data"):
            date_str = single_date.strftime("%Y%m%d")
            url = f"{base_url}{date_str}.export.CSV.zip"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    # Read CSV with tab separator and no header
                    df = pd.read_csv(url, compression='zip', sep='\t', header=None, low_memory=False)
                    
                    # Column index for average tone
                    avgtone_index = 34
                    
                    # Calculate average tone
                    avg_tone = df[avgtone_index].mean()
                    
                    sentiment_data.append({
                        "date": single_date,
                        "avg_tone": avg_tone
                    })
                    logger.info(f"Successfully downloaded GDELT data for {date_str}")
                else:
                    logger.warning(f"Data not found for {date_str}")
            except Exception as e:
                logger.error(f"Error downloading GDELT data for {date_str}: {e}")
                
        if not sentiment_data:
            logger.warning("No GDELT data found for the specified date range")
            return pd.DataFrame()
            
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_df.set_index('date', inplace=True)
        
        return sentiment_df
        
    def collect_sentiment_data(self, start_date: Optional[str] = None, days: int = 60) -> pd.DataFrame:
        """Collect GDELT sentiment data.
        
        Args:
            start_date: Optional start date in YYYYMMDD format
            days: Number of days of historical data to download (default: 60)
            
        Returns:
            DataFrame with daily sentiment data
        """
        # If no start_date provided, use 'days' ago
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
            
        # Always use the raw file for incremental updates
        raw_path = self.data_dir / "market_features" / "gdelt_raw.parquet"
        existing_data = pd.DataFrame()
        if raw_path.exists():
            existing_data = pd.read_parquet(raw_path)
            if not existing_data.empty:
                existing_data.index = pd.to_datetime(existing_data.index)
                logger.info(f"Found existing raw GDELT data through {existing_data.index.max().date()}")
        
        # Determine date range for new data
        if not existing_data.empty:
            # Start from the day after the latest existing data
            new_start = (existing_data.index.max() + timedelta(days=1)).strftime("%Y%m%d")
            if new_start > datetime.now().strftime("%Y%m%d"):
                logger.info("Raw GDELT data is up to date")
                return existing_data
        else:
            new_start = start_date
            
        # Fetch only new data
        new_data = self.fetch_gdelt_data(new_start, datetime.now().strftime("%Y%m%d"))
        
        if new_data.empty:
            return existing_data
            
        # Combine with existing data
        if not existing_data.empty:
            combined_data = pd.concat([existing_data, new_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data = combined_data.sort_index()
            combined_data.to_parquet(raw_path)
            return combined_data
        else:
            new_data.to_parquet(raw_path)
            return new_data 
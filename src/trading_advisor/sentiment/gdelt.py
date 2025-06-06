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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

logger = logging.getLogger(__name__)

class GDELTClient:
    """Client for fetching GDELT sentiment data."""
    
    def __init__(self, data_dir: Path):
        """Initialize GDELT client.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.base_url = "http://data.gdeltproject.org/gdeltv2/"
        self.masterfile_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
        
    def get_available_timestamps(self, date: str) -> List[str]:
        """Get available timestamps for a given date.
        
        Args:
            date: Date in YYYYMMDD format
            
        Returns:
            List of available timestamps for the date
        """
        try:
            response = requests.get(self.masterfile_url)
            if response.status_code == 200:
                lines = response.text.split('\n')
                timestamps = []
                for line in lines:
                    if line.strip() and '.export.CSV.zip' in line:
                        timestamp = line.split('/')[-1].split('.')[0]
                        if timestamp.startswith(date):
                            timestamps.append(timestamp)
                return sorted(timestamps)
        except Exception as e:
            logger.error(f"Error getting available timestamps for {date}: {e}")
        return []
        
    def fetch_gdelt_data(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch GDELT sentiment data for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format (default: today)
            
        Returns:
            DataFrame with sentiment data
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        if end_date is None:
            end = pd.to_datetime('today')
        else:
            end = pd.to_datetime(end_date)
            
        # Generate date range
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        # Collect sentiment data
        sentiment_data = []
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
            task = progress.add_task("Fetching GDELT data...", total=len(date_range))
            for single_date in date_range:
                date_str = single_date.strftime("%Y%m%d")
                
                # Get available timestamps for this date
                timestamps = self.get_available_timestamps(date_str)
                if not timestamps:
                    logger.warning(f"No data available for {date_str}")
                    progress.update(task, advance=1)
                    continue
                
                # Use the latest timestamp for the day
                latest_timestamp = timestamps[-1]
                url = f"{self.base_url}{latest_timestamp}.export.CSV.zip"
                
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
                progress.update(task, advance=1)
                time.sleep(2)  # Be nice to the GDELT server
                
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
            raw_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists
            combined_data['date'] = combined_data.index
            combined_data.to_parquet(raw_path)
            return combined_data
        else:
            raw_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists
            new_data['date'] = new_data.index
            new_data.to_parquet(raw_path)
            return new_data 
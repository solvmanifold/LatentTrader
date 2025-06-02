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

logger = logging.getLogger(__name__)

class GDELTClient:
    """Client for collecting GDELT news sentiment data."""
    
    def __init__(self, data_dir: Path):
        """Initialize the GDELT client.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.sentiment_dir = data_dir / "market_features" / "sentiment" / "gdelt"
        self.sentiment_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_gdelt_url(self, date: str) -> str:
        """Get the GDELT URL for a specific date.
        
        Args:
            date: Date in YYYYMMDD format
            
        Returns:
            URL for GDELT data
        """
        return f"http://data.gdeltproject.org/gdeltv2/{date}.export.CSV.zip"
        
    def _download_gdelt_data(self, date: str) -> Optional[pd.DataFrame]:
        """Download GDELT data for a specific date.
        
        Args:
            date: Date in YYYYMMDD format
            
        Returns:
            DataFrame with GDELT data or None if download fails
        """
        url = self._get_gdelt_url(date)
        try:
            # Download and read the CSV
            df = pd.read_csv(url, compression='zip', header=None, 
                           names=['GLOBALEVENTID', 'SQLDATE', 'MonthYear', 'Year', 'FractionDate',
                                 'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode',
                                 'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code',
                                 'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
                                 'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode',
                                 'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code',
                                 'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
                                 'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode',
                                 'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources',
                                 'NumArticles', 'AvgTone', 'Actor1Geo_Type', 'Actor1Geo_FullName',
                                 'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 'Actor1Geo_ADM2Code',
                                 'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID',
                                 'Actor2Geo_Type', 'Actor2Geo_FullName', 'Actor2Geo_CountryCode',
                                 'Actor2Geo_ADM1Code', 'Actor2Geo_ADM2Code', 'Actor2Geo_Lat',
                                 'Actor2Geo_Long', 'Actor2Geo_FeatureID', 'ActionGeo_Type',
                                 'ActionGeo_FullName', 'ActionGeo_CountryCode', 'ActionGeo_ADM1Code',
                                 'ActionGeo_ADM2Code', 'ActionGeo_Lat', 'ActionGeo_Long',
                                 'ActionGeo_FeatureID', 'DATEADDED', 'SOURCEURL'])
            
            # Filter for financial news
            df = df[df['EventCode'].str.startswith('14')]  # Financial events
            
            # Extract relevant columns
            df = df[['SQLDATE', 'AvgTone', 'NumMentions', 'NumSources', 'NumArticles', 'GoldsteinScale']]
            
            # Convert date
            df['SQLDATE'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading GDELT data for {date}: {e}")
            return None
            
    def collect_sentiment_data(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Collect GDELT sentiment data for a date range.
        
        Args:
            start_date: Optional start date for data collection (YYYY-MM-DD)
            
        Returns:
            DataFrame with aggregated sentiment metrics
        """
        if start_date:
            start = pd.to_datetime(start_date)
        else:
            start = pd.to_datetime('today') - pd.Timedelta(days=30)
            
        end = pd.to_datetime('today')
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        all_data = []
        for date in date_range:
            date_str = date.strftime('%Y%m%d')
            df = self._download_gdelt_data(date_str)
            if df is not None:
                all_data.append(df)
            time.sleep(1)  # Be nice to GDELT servers
            
        if not all_data:
            return pd.DataFrame()
            
        # Combine all data
        combined_df = pd.concat(all_data)
        
        # Aggregate by date
        daily_sentiment = combined_df.groupby('SQLDATE').agg({
            'AvgTone': 'mean',
            'NumMentions': 'sum',
            'NumSources': 'sum',
            'NumArticles': 'sum',
            'GoldsteinScale': 'mean'
        }).reset_index()
        
        # Calculate additional metrics
        daily_sentiment['sentiment_score'] = daily_sentiment['AvgTone'] / 100  # Normalize to [-1, 1]
        daily_sentiment['news_volume'] = daily_sentiment['NumArticles']
        daily_sentiment['news_impact'] = daily_sentiment['NumMentions'] * daily_sentiment['sentiment_score']
        
        # Calculate rolling metrics
        daily_sentiment['sentiment_ma5'] = daily_sentiment['sentiment_score'].rolling(5).mean()
        daily_sentiment['sentiment_ma20'] = daily_sentiment['sentiment_score'].rolling(20).mean()
        daily_sentiment['volume_ma5'] = daily_sentiment['news_volume'].rolling(5).mean()
        daily_sentiment['volume_ma20'] = daily_sentiment['news_volume'].rolling(20).mean()
        
        # Save to parquet
        output_file = self.sentiment_dir / "gdelt_sentiment.parquet"
        daily_sentiment.to_parquet(output_file)
        logger.info(f"Saved GDELT sentiment data to {output_file}")
        
        return daily_sentiment 
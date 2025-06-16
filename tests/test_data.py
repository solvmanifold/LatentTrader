import os
import tempfile
import shutil
import unittest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from trading_advisor.data import download_stock_data, standardize_columns_and_date
from trading_advisor.market_breadth import calculate_market_breadth
from trading_advisor.sector_performance import calculate_sector_performance
from trading_advisor.sentiment import MarketSentiment
from trading_advisor.volatility import MarketVolatility
from trading_advisor.market_features import MarketFeatures
import numpy as np

class TestDataDownload(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.features_dir = Path(self.temp_dir) / "ticker_features"
        self.features_dir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_download_stock_data(self):
        ticker = "AAPL"
        df = download_stock_data(ticker, features_dir=self.features_dir)
        # Ensure columns are standardized (lowercase) before any further processing
        df = standardize_columns_and_date(df)
        self.assertFalse(df.empty)
        self.assertTrue((self.features_dir / f"{ticker}_features.parquet").exists())
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))

    def test_long_term_moving_averages(self):
        """Test that long-term moving averages are calculated correctly."""
        ticker = "AAPL"
        # Use a smaller window for testing (50 days instead of 200)
        df = download_stock_data(ticker, features_dir=self.features_dir, history_days=100)
        # Ensure columns are standardized (lowercase) before any further processing
        df = standardize_columns_and_date(df)
        self.assertFalse(df.empty)
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
        
        # Check that we have enough data
        self.assertGreaterEqual(len(df), 50)
        
        # Verify moving average columns exist and have values
        ma_columns = ['sma_20', 'sma_50']  # Using smaller windows for testing
        for col in ma_columns:
            self.assertIn(col, df.columns)
            # Check that we have non-NaN values for the last 10 rows
            last_values = df[col].tail(10)
            self.assertFalse(last_values.isna().any(), f"Found NaN values in {col}")
            
        # Verify the moving averages are calculated correctly
        for col in ma_columns:
            window = int(col.split('_')[1])
            expected_ma = df['close'].rolling(window=window).mean()
            pd.testing.assert_series_equal(df[col], expected_ma, check_names=False)

    def test_future_data_handling(self):
        """Test handling of future data by simulating a gap in the data."""
        ticker = "AAPL"
        # Use a unique features filename for this test
        features_filename = f"{ticker}_test_future_data.parquet"
        
        # First download with enough history for technical indicators
        df1 = download_stock_data(ticker, features_dir=self.features_dir, history_days=100, features_filename=features_filename)
        # Ensure columns are standardized (lowercase) before any further processing
        df1 = standardize_columns_and_date(df1)
        self.assertFalse(df1.empty)
        self.assertTrue(isinstance(df1.index, pd.DatetimeIndex))
        
        # Get the last 3 dates from the actual data
        last_dates = df1.index[-3:]
        
        # Delete the last 3 rows from the saved file
        features_path = self.features_dir / features_filename
        df_truncated = df1.iloc[:-3]
        df_truncated.to_parquet(features_path)
        
        # Download again with fewer retries
        df2 = download_stock_data(ticker, features_dir=self.features_dir, max_retries=1, features_filename=features_filename)
        # Ensure columns are standardized (lowercase) before any further processing
        df2 = standardize_columns_and_date(df2)
        self.assertTrue(isinstance(df2.index, pd.DatetimeIndex))
        
        # Only compare raw columns
        raw_cols = [col for col in ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits'] if col in df1.columns]
        for date in last_dates:
            self.assertIn(date, df2.index)
            # Use rtol and atol for floating point comparison
            pd.testing.assert_series_equal(
                df1.loc[date, raw_cols],
                df2.loc[date, raw_cols],
                check_names=False,
                check_index=False,
                rtol=1e-3,  # 0.1% relative tolerance
                atol=1e-3   # 0.001 absolute tolerance
            )

    def test_calculate_market_breadth(self):
        ticker = "AAPL"
        df = download_stock_data(ticker, features_dir=self.features_dir)
        # Ensure we have the required columns in lowercase
        df = standardize_columns_and_date(df)
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
        breadth_df = calculate_market_breadth(df)
        self.assertFalse(breadth_df.empty)
        # Verify we have the expected breadth indicators
        expected_columns = ['daily_breadth_adv_dec_line', 'daily_breadth_new_highs', 
                          'daily_breadth_new_lows', 'daily_breadth_above_ma20', 
                          'daily_breadth_above_ma50', 'daily_breadth_rsi_bullish', 
                          'daily_breadth_rsi_oversold', 'daily_breadth_rsi_overbought', 
                          'daily_breadth_macd_bullish']
        for col in expected_columns:
            self.assertIn(col, breadth_df.columns)

    def test_calculate_sector_performance(self):
        ticker = "AAPL"
        df = download_stock_data(ticker, features_dir=self.features_dir)
        # Ensure we have the required columns in lowercase
        df = standardize_columns_and_date(df)
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
        df['sector'] = 'Technology'  # Ensure 'sector' column exists
        sector_dfs = calculate_sector_performance(df, self.features_dir)
        self.assertIn('all_sectors', sector_dfs)
        # Verify the sector performance DataFrame has the expected columns
        sector_df = sector_dfs['all_sectors']
        expected_columns = ['Technology_sector_performance_returns_1d', 
                          'Technology_sector_performance_momentum_5d', 
                          'Technology_sector_performance_relative_strength']
        for col in expected_columns:
            self.assertIn(col, sector_df.columns)

    def test_generate_sentiment_features(self):
        """Test sentiment feature generation."""
        # Initialize market features with test directory
        market_features = MarketFeatures(str(self.temp_dir))
        
        # Create test GDELT data for a historical date range
        # Use a date range from 2023 since today is June 16, 2025
        start_date = pd.Timestamp('20230101')
        dates = pd.date_range(start=start_date, periods=5, freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'avg_tone': np.random.randn(len(dates))
        })
        test_data.set_index('date', inplace=True)
        
        # Save test data
        gdelt_path = Path(self.temp_dir) / "market_features" / "gdelt_raw.parquet"
        gdelt_path.parent.mkdir(parents=True, exist_ok=True)
        test_data.to_parquet(gdelt_path)
        
        # Create a test ticker file to ensure MarketFeatures has data to work with
        ticker_dir = Path(self.temp_dir) / "ticker_features"
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ticker data with enough history for technical indicators
        # We need at least 50 days for technical indicators
        ticker_dates = pd.date_range(start=start_date - pd.Timedelta(days=60), periods=65, freq='D')
        ticker_data = pd.DataFrame({
            'date': ticker_dates,
            'open': 100 + np.random.randn(len(ticker_dates)),
            'high': 101 + np.random.randn(len(ticker_dates)),
            'low': 99 + np.random.randn(len(ticker_dates)),
            'close': 100 + np.random.randn(len(ticker_dates)),
            'volume': 1000000 + np.random.randint(0, 1000000, len(ticker_dates))
        })
        ticker_data.set_index('date', inplace=True)
        
        # Calculate technical indicators
        from trading_advisor.analysis import calculate_technical_indicators
        ticker_data = calculate_technical_indicators(ticker_data)
        
        # Save the ticker data with technical indicators
        ticker_data.to_parquet(ticker_dir / "AAPL_features.parquet")
        
        # Generate market features (which includes sentiment)
        market_features.generate_market_features(days=5)
        
        # Load and validate sentiment features
        sentiment_path = Path(self.temp_dir) / "market_features" / "market_sentiment.parquet"
        self.assertTrue(sentiment_path.exists(), "Sentiment features file should exist")
        
        sentiment_df = pd.read_parquet(sentiment_path)
        self.assertFalse(sentiment_df.empty, "Sentiment DataFrame should not be empty")
        self.assertTrue(isinstance(sentiment_df.index, pd.DatetimeIndex), "Index should be DatetimeIndex")
        
        # Check for expected sentiment columns
        expected_columns = ['market_sentiment_ma5', 'market_sentiment_ma20', 'market_sentiment_momentum',
            'market_sentiment_volatility', 'market_sentiment_zscore']
        for col in expected_columns:
            self.assertIn(col, sentiment_df.columns, f"Expected column {col} not found")

    def test_generate_volatility_features(self):
        ticker = "AAPL"
        df = download_stock_data(ticker, features_dir=self.features_dir)
        # Ensure columns are standardized (lowercase) before any further processing
        df = standardize_columns_and_date(df)
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
        df['ticker'] = ticker  # Ensure 'ticker' column exists for volatility calculation
        volatility = MarketVolatility(Path(self.temp_dir))
        volatility_df = volatility.generate_volatility_features(df)
        self.assertFalse(volatility_df.empty)

    def test_standardize_columns_and_date(self):
        # Test with mixed case and spaces
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5),
            'Open Price': [100, 101, 102, 103, 104],
            'Close Price': [101, 102, 103, 104, 105],
            'Volume (Shares)': [1000, 2000, 3000, 4000, 5000]
        })
        
        # Test without source prefix
        standardized_df = standardize_columns_and_date(df)
        self.assertIn('open_price', standardized_df.columns)
        self.assertIn('close_price', standardized_df.columns)
        self.assertIn('volume_shares', standardized_df.columns)
        self.assertTrue(isinstance(standardized_df.index, pd.DatetimeIndex))
        self.assertTrue(all(standardized_df.index.hour == 0))  # All dates should be normalized
        
        # Test with special characters
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5),
            'Open@Price': [100, 101, 102, 103, 104],
            'Close-Price': [101, 102, 103, 104, 105],
            'Volume%Change': [1000, 2000, 3000, 4000, 5000]
        })
        
        standardized_df = standardize_columns_and_date(df)
        self.assertIn('open_price', standardized_df.columns)
        self.assertIn('close_price', standardized_df.columns)
        self.assertIn('volume_change', standardized_df.columns)
        self.assertTrue(isinstance(standardized_df.index, pd.DatetimeIndex))
        self.assertTrue(all(standardized_df.index.hour == 0))  # All dates should be normalized

    def test_standardize_columns_with_special_chars(self):
        # Test with special characters in column names
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=3),
            'Open@Price': [100, 101, 102],
            'Close-Price': [101, 102, 103],
            'Volume%Change': [0.1, 0.2, 0.3]
        })
        
        standardized_df = standardize_columns_and_date(df)
        self.assertIn('open_price', standardized_df.columns)
        self.assertIn('close_price', standardized_df.columns)
        self.assertIn('volume_change', standardized_df.columns)
        self.assertTrue(isinstance(standardized_df.index, pd.DatetimeIndex))
        self.assertTrue(all(standardized_df.index.hour == 0))  # All dates should be normalized

if __name__ == '__main__':
    unittest.main() 
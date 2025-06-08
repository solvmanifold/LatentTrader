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
        df = standardize_columns_and_date(df, keep_date_column=True)
        self.assertFalse(df.empty)
        self.assertTrue((self.features_dir / f"{ticker}_features.parquet").exists())

    def test_long_term_moving_averages(self):
        """Test that long-term moving averages are calculated correctly."""
        ticker = "AAPL"
        # Use a smaller window for testing (50 days instead of 200)
        df = download_stock_data(ticker, features_dir=self.features_dir, history_days=100)
        # Ensure columns are standardized (lowercase) before any further processing
        df = standardize_columns_and_date(df, keep_date_column=True)
        self.assertFalse(df.empty)
        
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
        df1 = standardize_columns_and_date(df1, keep_date_column=True)
        self.assertFalse(df1.empty)
        
        # Get the last 3 dates from the actual data
        last_dates = df1.index[-3:]
        
        # Delete the last 3 rows from the saved file
        features_path = self.features_dir / features_filename
        df_truncated = df1.iloc[:-3]
        df_truncated.to_parquet(features_path)
        
        # Download again with fewer retries
        df2 = download_stock_data(ticker, features_dir=self.features_dir, max_retries=1, features_filename=features_filename)
        # Ensure columns are standardized (lowercase) before any further processing
        df2 = standardize_columns_and_date(df2, keep_date_column=True)
        
        # Only compare raw columns
        raw_cols = [col for col in ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits'] if col in df1.columns]
        for date in last_dates:
            self.assertIn(date, df2.index)
            pd.testing.assert_series_equal(
                df1.loc[date, raw_cols],
                df2.loc[date, raw_cols],
                check_names=False,
                check_index=False
            )

    def test_calculate_market_breadth(self):
        ticker = "AAPL"
        df = download_stock_data(ticker, features_dir=self.features_dir)
        # Ensure we have the required columns in lowercase
        df = standardize_columns_and_date(df, keep_date_column=True)
        breadth_df = calculate_market_breadth(df)
        self.assertFalse(breadth_df.empty)
        # Verify we have the expected breadth indicators
        expected_columns = ['adv_dec_line', 'new_highs', 'new_lows', 'above_ma20', 
                          'above_ma50', 'rsi_bullish', 'rsi_oversold', 'rsi_overbought', 
                          'macd_bullish']
        for col in expected_columns:
            self.assertIn(col, breadth_df.columns)

    def test_calculate_sector_performance(self):
        ticker = "AAPL"
        df = download_stock_data(ticker, features_dir=self.features_dir)
        # Ensure we have the required columns in lowercase
        df = standardize_columns_and_date(df, keep_date_column=True)
        df['sector'] = 'Technology'  # Ensure 'sector' column exists
        sector_dfs = calculate_sector_performance(df, self.features_dir)
        self.assertIn('all_sectors', sector_dfs)
        # Verify the sector performance DataFrame has the expected columns
        sector_df = sector_dfs['all_sectors']
        expected_columns = ['Technology_returns_1d', 'Technology_momentum_5d', 'Technology_relative_strength']
        for col in expected_columns:
            self.assertIn(col, sector_df.columns)

    def test_generate_sentiment_features(self):
        """Test sentiment feature generation."""
        # Initialize sentiment analyzer with test directory
        sentiment = MarketSentiment(Path(self.temp_dir))
        
        # Generate sentiment features with a small date range
        sentiment_df = sentiment.generate_sentiment_features(days=5)
        
        # Basic validation
        self.assertFalse(sentiment_df.empty)
        self.assertTrue(isinstance(sentiment_df.index, pd.DatetimeIndex))
        # Check for expected sentiment columns
        expected_columns = ['sentiment_ma5', 'sentiment_ma20', 'sentiment_momentum', 
                          'sentiment_volatility', 'sentiment_zscore']
        for col in expected_columns:
            self.assertIn(col, sentiment_df.columns)

    def test_generate_volatility_features(self):
        ticker = "AAPL"
        df = download_stock_data(ticker, features_dir=self.features_dir)
        # Ensure columns are standardized (lowercase) before any further processing
        df = standardize_columns_and_date(df, keep_date_column=True)
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
        
        # Test without source prefix, keeping date column
        standardized_df = standardize_columns_and_date(df, keep_date_column=True)
        self.assertIn('date', standardized_df.columns)
        self.assertIn('open_price', standardized_df.columns)
        self.assertIn('close_price', standardized_df.columns)
        self.assertIn('volume_shares', standardized_df.columns)
        self.assertEqual(standardized_df.columns[0], 'date')
        
        # Test with source prefix, keeping date column
        standardized_df = standardize_columns_and_date(df, source_prefix='market', keep_date_column=True)
        self.assertIn('market_open_price', standardized_df.columns)
        self.assertIn('market_close_price', standardized_df.columns)
        self.assertIn('market_volume_shares', standardized_df.columns)
        
        # Test date handling
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(standardized_df['date']))
        self.assertTrue(all(standardized_df['date'].dt.hour == 0))  # All dates should be normalized
        
        # Test without keeping date column
        standardized_df = standardize_columns_and_date(df)
        self.assertNotIn('date', standardized_df.columns)
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

if __name__ == '__main__':
    unittest.main() 
import os
import tempfile
import shutil
import unittest
import pandas as pd
from pathlib import Path
from trading_advisor.data import download_stock_data
from trading_advisor.market_breadth import calculate_market_breadth
from trading_advisor.sector_performance import calculate_sector_performance
from trading_advisor.sentiment import MarketSentiment
from trading_advisor.volatility import MarketVolatility

class TestDataDownload(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.features_dir = Path(self.temp_dir) / "features"
        self.features_dir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_download_stock_data(self):
        ticker = "AAPL"
        df = download_stock_data(ticker, features_dir=self.features_dir)
        self.assertFalse(df.empty)
        self.assertTrue((self.features_dir / f"{ticker}_features.parquet").exists())

    def test_calculate_market_breadth(self):
        ticker = "AAPL"
        df = download_stock_data(ticker, features_dir=self.features_dir)
        breadth_df = calculate_market_breadth(df)
        self.assertFalse(breadth_df.empty)

    def test_calculate_sector_performance(self):
        ticker = "AAPL"
        df = download_stock_data(ticker, features_dir=self.features_dir)
        df['sector'] = 'Technology'  # Ensure 'sector' column exists
        sector_dfs = calculate_sector_performance(df, self.features_dir)
        self.assertIn('all_sectors', sector_dfs)

    def test_generate_sentiment_features(self):
        sentiment = MarketSentiment(Path(self.temp_dir))
        sentiment_df = sentiment.generate_sentiment_features()
        self.assertFalse(sentiment_df.empty)

    def test_generate_volatility_features(self):
        ticker = "AAPL"
        df = download_stock_data(ticker, features_dir=self.features_dir)
        df['ticker'] = ticker  # Ensure 'ticker' column exists for volatility calculation
        volatility = MarketVolatility(Path(self.temp_dir))
        volatility_df = volatility.generate_volatility_features(df)
        self.assertFalse(volatility_df.empty)

if __name__ == '__main__':
    unittest.main() 
import glob
import pandas as pd
from pathlib import Path
from trading_advisor.sector_mapping import load_sector_mapping

data_dir = Path('data')
market_features_dir = data_dir / 'market_features'
features_dir = data_dir / 'features'

# Load sector mapping
sector_mapping = load_sector_mapping(str(market_features_dir))

# Get all tickers
feature_files = list(features_dir.glob('*_features.parquet'))
ticker_list = [f.stem.replace('_features', '') for f in feature_files]

ticker_df = pd.DataFrame()
for ticker in ticker_list:
    features_path = features_dir / f"{ticker}_features.parquet"
    if not features_path.exists():
        continue
    df = pd.read_parquet(features_path)
    df['ticker'] = ticker
    df['sector'] = sector_mapping.get(ticker, 'Unknown')
    ticker_df = pd.concat([ticker_df, df])

print('ticker_df info:')
print(ticker_df.info())
print(ticker_df.head()) 
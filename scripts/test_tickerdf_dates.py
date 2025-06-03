import glob
import pandas as pd
from pathlib import Path
from trading_advisor.sector_mapping import load_sector_mapping

# Mimic CLI data loading

data_dir = Path('data')
market_features_dir = data_dir / 'market_features'
features_dir = data_dir / 'features'

sector_mapping = load_sector_mapping(str(market_features_dir))
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

print('ticker_df min date:', ticker_df.index.min())
print('ticker_df max date:', ticker_df.index.max())

# Simulate CLI start_date logic (None means use all data)
start_date = None
print('start_date used by CLI:', start_date) 
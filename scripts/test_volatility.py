import glob
import pandas as pd
from trading_advisor.volatility import calculate_market_volatility

# Combine all ticker data
feature_files = glob.glob('data/features/*_features.parquet')
dfs = []
for f in feature_files:
    df = pd.read_parquet(f)
    df['ticker'] = f.split('/')[-1].replace('_features.parquet', '')
    dfs.append(df)

if not dfs:
    print('No ticker data found!')
    exit(1)

combined_df = pd.concat(dfs)
combined_df.index = pd.to_datetime(combined_df.index)
print('Combined DataFrame info:')
print(combined_df.info())
print(combined_df.head())

# Run volatility calculation
vol_df = calculate_market_volatility(combined_df)
print('Volatility DataFrame info:')
print(vol_df.info())
print(vol_df.head()) 
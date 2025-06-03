import pandas as pd
import glob
import os

files = glob.glob('data/market_features/*.parquet')
for f in files:
    try:
        df = pd.read_parquet(f)
        if not df.empty:
            last = df.tail(1)
            nans = last.isna().sum(axis=1).iloc[0]
            total = last.shape[1]
            if nans > 0:
                print(f'{os.path.basename(f)}: {nans}/{total} NaNs in last row')
            else:
                print(f'{os.path.basename(f)}: 0 NaNs in last row')
        else:
            print(f'{os.path.basename(f)}: EMPTY')
    except Exception as e:
        print(f'{os.path.basename(f)}: ERROR ({e})') 
import pandas as pd
import json
from trading_advisor.analysis import calculate_technical_indicators, get_analyst_targets
import subprocess

PARQUET_PATH = "features/AAPL_features.parquet"
N_DELETE = 10  # Number of rows to delete from the end
TICKERS_FILE = "tickers.txt"

# 1. Load the Parquet file
df = pd.read_parquet(PARQUET_PATH)
print(f"Original length: {len(df)}")

# 2. Delete the last N rows
df_trunc = df.iloc[:-N_DELETE]
print(f"After deletion: {len(df_trunc)}")

# 3. Save the truncated file
df_trunc.to_parquet(PARQUET_PATH)
print("Truncated file saved.")

# 4. Call the CLI to update/fill in missing data
print("Running: trading-advisor init-features --tickers tickers.txt")
result = subprocess.run([
    "trading-advisor", "init-features", "--tickers", TICKERS_FILE
], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# 5. Reload and inspect the Parquet file
df_updated = pd.read_parquet(PARQUET_PATH)
print(f"After CLI update, length: {len(df_updated)}")
print(df_updated.tail(10))

# 6. Simulate a pipeline update: merge with new data (simulate by reusing the original last N rows)
# In a real update, you'd download new data for the missing dates.
# For this test, we'll just use the deleted rows as 'new data'
df_new = df.iloc[-N_DELETE:]
df_merged = pd.concat([df_trunc, df_new])
df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
df_merged = df_merged.sort_index()

try:
    # 7. Calculate indicators and scores on the merged DataFrame
    df_features = calculate_technical_indicators(df_merged)
    # 8. Add analyst_targets to the last row
    analyst_targets = get_analyst_targets("AAPL")
    if analyst_targets and not df_features.empty:
        df_features.at[df_features.index[-1], 'analyst_targets'] = json.dumps(analyst_targets)
    print("Feature calculation successful! Final length:", len(df_features))
except Exception as e:
    print("Error during feature calculation:", e) 
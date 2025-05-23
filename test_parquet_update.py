import pandas as pd
import json
from trading_advisor.analysis import calculate_technical_indicators, get_analyst_targets
import subprocess
import tempfile
import os
from pathlib import Path

def test_parquet_update():
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        PARQUET_PATH = temp_dir / "AAPL_features.parquet"
        N_DELETE = 10  # Number of rows to delete from the end
        TICKERS_FILE = temp_dir / "tickers.txt"
        
        # Create a sample tickers file
        with open(TICKERS_FILE, 'w') as f:
            f.write("AAPL\n")
        
        # Create a sample DataFrame
        df = pd.DataFrame({
            'Open': [100.0] * 1257,
            'High': [105.0] * 1257,
            'Low': [95.0] * 1257,
            'Close': [102.0] * 1257,
            'Volume': [1000000] * 1257
        }, index=pd.date_range(start='2020-01-01', periods=1257, freq='B'))
        
        # Save initial DataFrame
        df.to_parquet(PARQUET_PATH)
        print(f"Original length: {len(df)}")
        
        # Delete the last N rows
        df_trunc = df.iloc[:-N_DELETE]
        print(f"After deletion: {len(df_trunc)}")
        
        # Save the truncated file
        df_trunc.to_parquet(PARQUET_PATH)
        print("Truncated file saved.")
        
        # Call the CLI to update/fill in missing data
        print("Running: trading-advisor init-features --tickers tickers.txt")
        result = subprocess.run([
            "trading-advisor", "init-features", 
            "--tickers", str(TICKERS_FILE),
            "--features-dir", str(temp_dir)
        ], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Reload and inspect the Parquet file
        df_updated = pd.read_parquet(PARQUET_PATH)
        print(f"After CLI update, length: {len(df_updated)}")
        print(df_updated.tail(10))
        
        # Simulate a pipeline update: merge with new data
        df_new = df.iloc[-N_DELETE:]
        df_merged = pd.concat([df_trunc, df_new])
        df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
        df_merged = df_merged.sort_index()
        
        try:
            # Calculate indicators and scores on the merged DataFrame
            df_features = calculate_technical_indicators(df_merged)
            # Add analyst_targets to the last row
            analyst_targets = get_analyst_targets("AAPL")
            if analyst_targets and not df_features.empty:
                df_features.at[df_features.index[-1], 'analyst_targets'] = json.dumps(analyst_targets)
            print("Feature calculation successful! Final length:", len(df_features))
        except Exception as e:
            print("Error during feature calculation:", e)

if __name__ == "__main__":
    test_parquet_update() 
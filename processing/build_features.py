import os
import sys

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: {e}. Please ensure you are using the correct environment (e.g., .venv).")
    sys.exit(1)

def build_features():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming the structure is project_root/processing/build_features.py
    # and data is in project_root/data/
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'crude_5m.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading data from {csv_path}...")
    # Load data (skipping multi-line headers as discovered in notebook)
    df = pd.read_csv(csv_path, skiprows=3, names=['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume'])
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    print("Calculating technical features...")
    # Calculate features
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['hl_range'] = df['High'] - df['Low']
    df['co_range'] = df['Close'] - df['Open']
    df['volatility'] = df['log_ret'].rolling(window=12).std()
    df['volume_pct'] = df['Volume'].pct_change()

    # Handle inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop NaNs
    df_features = df[['log_ret', 'hl_range', 'co_range', 'volatility', 'volume_pct']].dropna()
    
    output_path = os.path.join(project_root, 'data', 'ohlcv_features.csv')
    df_features.to_csv(output_path)
    print(f"Success: Features built and saved to {output_path}")
    print(f"Output Head:\n{df_features.head()}")

if __name__ == "__main__":
    build_features()

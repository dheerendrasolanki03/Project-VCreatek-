import yfinance as yf

# Crude Oil Futures
symbol = "CL=F"

print(f"Downloading data for {symbol}...")
data = yf.download(
    tickers=symbol,
    interval="5m",     
    period="1mo", 
    progress=False
)

output_file = "data/crude_5m.csv"
data.to_csv(output_file)
print(f"Saved {output_file} with shape:", data.shape)

# User's debug info
print("\nData Info:")
data.info()
print("\nData Summary:")
print(data.describe())

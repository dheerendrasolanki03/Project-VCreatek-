import websocket
import json
import csv
import os
from datetime import datetime

SOCKET = "wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

output_file = os.path.join(DATA_DIR, "raw_orderbook.csv")

# Always (re)create file with header if not exists
if not os.path.isfile(output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "best_bid_price", "best_bid_qty",
            "best_ask_price", "best_ask_qty",
            "bid_vol_sum", "ask_vol_sum"
        ])

def on_message(ws, message):
    data = json.loads(message)

    bids = data["bids"]
    asks = data["asks"]

    row = [
        datetime.utcnow().timestamp(),
        float(bids[0][0]), float(bids[0][1]),
        float(asks[0][0]), float(asks[0][1]),
        sum(float(b[1]) for b in bids),
        sum(float(a[1]) for a in asks)
    ]

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print("Recorded:", row[0])

def on_open(ws):
    print("Connected to Binance depth stream")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET,
                                 on_open=on_open,
                                 on_message=on_message)
    ws.run_forever()

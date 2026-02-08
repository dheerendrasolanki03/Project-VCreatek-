import websocket
import json
import os
import numpy as np
from xgboost import XGBClassifier

# ===== Load trained model =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = XGBClassifier()
model.load_model(os.path.join(BASE_DIR, "xgb_hft_binary.json"))

# ===== Binance WebSocket =====
SOCKET = "wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms"

# ===== State variables =====
prev_mid = None
entry_price = None
position = None   # "LONG" or "SHORT"
total_pnl = 0
trade_count = 0

def on_message(ws, message):
    global prev_mid, entry_price, position, total_pnl, trade_count
    
    data = json.loads(message)
    bids = data["bids"]
    asks = data["asks"]

    best_bid_price = float(bids[0][0])
    best_bid_qty = float(bids[0][1])
    best_ask_price = float(asks[0][0])
    best_ask_qty = float(asks[0][1])

    bid_vol_sum = sum(float(b[1]) for b in bids)
    ask_vol_sum = sum(float(a[1]) for a in asks)

    # ===== Feature Engineering =====
    spread = best_ask_price - best_bid_price
    obi = (bid_vol_sum - ask_vol_sum) / (bid_vol_sum + ask_vol_sum)
    micro_price = (
        (best_ask_price * best_bid_qty + best_bid_price * best_ask_qty) /
        (best_bid_qty + best_ask_qty)
    )

    mid_price = (best_bid_price + best_ask_price) / 2

    # ===== Exit previous position after 1 tick =====
    if position is not None and prev_mid is not None:
        exit_mid = mid_price
        
        if position == "LONG":
            pnl = exit_mid - entry_price
        else:
            pnl = entry_price - exit_mid
        
        total_pnl += pnl
        trade_count += 1
        
        print(f"Exit {position} | PnL: {pnl:.5f} | Total PnL: {total_pnl:.5f}")
        
        position = None
        entry_price = None

    # ===== Build feature array for model =====
    X = np.array([[spread, obi, micro_price]])

    # ===== Model prediction =====
    prediction = model.predict(X)[0]  
    # 1 = UP → LONG, 0 = DOWN → SHORT

    # ===== Enter new trade =====
    if prediction == 1:
        position = "LONG"
        entry_price = best_ask_price
        print(f"BUY  @ {entry_price}")

    else:
        position = "SHORT"
        entry_price = best_bid_price
        print(f"SELL @ {entry_price}")

    prev_mid = mid_price

def on_open(ws):
    print("Live HFT Simulator Started...")
    print("Receiving live Binance order book data...\n")

def on_error(ws, error):
    print("Error:", error)

def on_close(ws):
    print("\nConnection closed")
    print("Total Trades:", trade_count)
    print("Final Total PnL:", round(total_pnl, 5))

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET,
                                 on_open=on_open,
                                 on_message=on_message,
                                 on_error=on_error,
                                 on_close=on_close)
    ws.run_forever()

import os
import json
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket
from xgboost import XGBClassifier
import websockets

# ===== Load trained model (500ms horizon) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "xgb_model_500ms.json")

model = XGBClassifier()
model.load_model(MODEL_PATH)

# ===== Binance WebSocket URL =====
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms"

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    print("Frontend connected.")

    # Trading state
    position = None
    entry_price = None
    prev_mid = None
    total_pnl = 0
    trade_count = 0

    async with websockets.connect(BINANCE_WS) as binance_ws:
        while True:
            message = await binance_ws.recv()
            data = json.loads(message)

            bids = data["bids"]
            asks = data["asks"]

            best_bid_price = float(bids[0][0])
            best_bid_qty = float(bids[0][1])
            best_ask_price = float(asks[0][0])
            best_ask_qty = float(asks[0][1])

            bid_vol_sum = sum(float(b[1]) for b in bids)
            ask_vol_sum = sum(float(a[1]) for a in asks)

            # ===== Features =====
            spread = best_ask_price - best_bid_price
            obi = (bid_vol_sum - ask_vol_sum) / (bid_vol_sum + ask_vol_sum)
            micro_price = (
                (best_ask_price * best_bid_qty + best_bid_price * best_ask_qty) /
                (best_bid_qty + best_ask_qty)
            )
            mid_price = (best_bid_price + best_ask_price) / 2

            # ===== Exit previous position after horizon =====
            last_trade_pnl = 0
            if position is not None and prev_mid is not None:
                if position == "LONG":
                    last_trade_pnl = mid_price - entry_price
                else:
                    last_trade_pnl = entry_price - mid_price

                total_pnl += last_trade_pnl
                trade_count += 1

            # ===== Prediction =====
            X = np.array([[spread, obi, micro_price]])
            prediction = int(model.predict(X)[0])

            if prediction == 1:
                position = "LONG"
                entry_price = best_ask_price
                action = "BUY"
                pred_text = "UP"
            else:
                position = "SHORT"
                entry_price = best_bid_price
                action = "SELL"
                pred_text = "DOWN"

            prev_mid = mid_price

            # ===== Send to frontend =====
            payload = {
                "best_bid": best_bid_price,
                "best_ask": best_ask_price,
                "spread": round(spread, 6),
                "obi": round(obi, 4),
                "prediction": pred_text,
                "action": action,
                "entry_price": entry_price,
                "last_trade_pnl": round(last_trade_pnl, 6),
                "total_pnl": round(total_pnl, 6),
                "trade_count": trade_count
            }

            await client_ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)

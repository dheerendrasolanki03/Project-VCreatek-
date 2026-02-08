import { useEffect, useState } from "react";

export default function App() {
  const [data, setData] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws");

    ws.onopen = () => setConnected(true);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      setData(msg);
    };

    ws.onclose = () => setConnected(false);

    return () => ws.close();
  }, []);

  return (
    <div className="bg-gray-900 text-white min-h-screen p-6">
      <h1 className="text-3xl font-bold mb-6">HFT Live Trading Dashboard</h1>

      {!connected && <p className="text-red-400">Connecting to backend...</p>}

      {data && (
        <div className="grid grid-cols-2 gap-6">

          <div className="bg-gray-800 p-4 rounded">
            <h2 className="text-xl mb-2">Market Data</h2>
            <p>Best Bid: {data.best_bid}</p>
            <p>Best Ask: {data.best_ask}</p>
            <p>Spread: {data.spread}</p>
            <p>OBI: {data.obi}</p>
          </div>

          <div className="bg-gray-800 p-4 rounded">
            <h2 className="text-xl mb-2">Model Prediction</h2>
            <p>Prediction: 
              <span className={data.prediction === "UP" ? "text-green-400" : "text-red-400"}>
                {" "}{data.prediction}
              </span>
            </p>
            <p>Action: {data.action}</p>
            <p>Entry Price: {data.entry_price}</p>
          </div>

          <div className="bg-gray-800 p-4 rounded">
            <h2 className="text-xl mb-2">Performance</h2>
            <p>Last Trade PnL: {data.last_trade_pnl}</p>
            <p>Total PnL: {data.total_pnl}</p>
            <p>Total Trades: {data.trade_count}</p>
          </div>

        </div>
      )}
    </div>
  );
}

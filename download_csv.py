#pip install ccxt pandas
# fetch_crypto.py — daily OHLCV -> a CSV the existing pipeline can read
import ccxt, pandas as pd

exchange = ccxt.binance()                 # or: ccxt.coinbase()
ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1w", limit=1000)
df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
df["date"] = pd.to_datetime(df["ts"], unit="ms")
df[["date", "open", "high", "low", "close", "volume"]].to_csv("crypto.csv", index=False)
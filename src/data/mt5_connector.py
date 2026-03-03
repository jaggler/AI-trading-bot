"""
MT5 Data Connector with caching
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle


class MT5Connector:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.connected = False

    def connect(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print("MT5 initialization failed")
            return False
        self.connected = True
        print(f"Connected to MT5: {mt5.terminal_info().name}")
        return True

    def disconnect(self):
        mt5.shutdown()
        self.connected = False

    def fetch_data(self, symbol, timeframe, start_date, end_date, cache=True):
        """
        Fetch historical data with caching
        """
        cache_file = self.data_dir / f"{symbol}_{timeframe}_{start_date}_{end_date}.pkl".replace(":", "-").replace(" ", "_")

        # Check cache
        if cache and cache_file.exists():
            print(f"Loading cached data: {cache_file}")
            return pd.read_pickle(cache_file)

        if not self.connected:
            self.connect()

        # Fetch from MT5
        rates = mt5.copy_rates_range(
            symbol,
            timeframe,
            start_date,
            end_date
        )

        if rates is None or len(rates) == 0:
            print(f"No data returned for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Cache if requested
        if cache:
            df.to_pickle(cache_file)
            print(f"Cached to: {cache_file}")

        return df

    def get_symbols(self):
        """Get all available symbols"""
        if not self.connected:
            self.connect()
        return [s.name for s in mt5.symbols_get()]


# Usage example
if __name__ == "__main__":
    conn = MT5Connector()
    conn.connect()

    # Fetch 5 years of EURUSD H1 data
    end = datetime.now()
    start = end - timedelta(days=365*5)

    df = conn.fetch_data("EURUSD", mt5.TIMEFRAME_H1, start, end)
    print(f"Fetched {len(df)} rows")
    print(df.head())

    conn.disconnect()

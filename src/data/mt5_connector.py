import MetaTrader5 as mt5
import pandas as pd

class MT5Connector:
    def __init__(self):
        pass

    def connect(self):
        """Connect to the MetaTrader 5 terminal."""
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            return False
        return True

    def get_rates(self, symbol, timeframe, num_bars):
        """Retrieve historical rates for a given symbol."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None:
            print(f"Failed to get rates for {symbol}, error code =", mt5.last_error())
            return None
        return pd.DataFrame(rates)

from src.data.mt5_connector import MT5Connector

def main():
    print("Initializing Forex ML Trading System...")

    connector = MT5Connector()
    if connector.connect():
        print("Successfully connected to MetaTrader 5.")

        # Example of getting rates (uncomment and replace parameters when running with real MT5 terminal)
        # import MetaTrader5 as mt5
        # df = connector.get_rates('EURUSD', mt5.TIMEFRAME_H1, 100)
        # if df is not None:
        #     print(f"Retrieved {len(df)} records.")
    else:
        print("Failed to connect to MetaTrader 5.")

if __name__ == "__main__":
    main()

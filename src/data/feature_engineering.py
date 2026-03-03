import pandas as pd
import pandas_ta as ta

def calculate_rsi(df, window=14, column='close'):
    """Calculate Relative Strength Index (RSI)."""
    return ta.rsi(df[column], length=window)

def calculate_ema(df, window=14, column='close'):
    """Calculate Exponential Moving Average (EMA)."""
    return ta.ema(df[column], length=window)

def calculate_fvg(df, high_col='high', low_col='low'):
    """Calculate Fair Value Gaps (FVG)."""
    # Simple placeholder logic for FVG calculation
    # Bullish FVG: Low of candle 3 > High of candle 1
    # Bearish FVG: High of candle 3 < Low of candle 1
    bullish_fvg = df[low_col].shift(-2) > df[high_col]
    bearish_fvg = df[high_col].shift(-2) < df[low_col]

    return pd.DataFrame({
        'bullish_fvg': bullish_fvg,
        'bearish_fvg': bearish_fvg
    })

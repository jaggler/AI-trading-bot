"""
Main execution script for forex ML trading system
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from datetime import datetime, timedelta
import yaml
import MetaTrader5 as mt5
import pandas as pd

from data.mt5_connector import MT5Connector
from data.feature_engineering import FeatureEngineer
from validation.walk_forward import WalkForwardValidator
from sklearn.ensemble import RandomForestClassifier


def run_pipeline(mode="research"):
    """
    Main pipeline execution

    mode: 'research' (backtest), 'train' (fit final model), 'live' (trading)
    """

    print(f"{'='*60}")
    print(f"FOREX ML TRADING SYSTEM - Mode: {mode.upper()}")
    print(f"{'='*60}")

    # 1. Data Collection
    print("\n[1] Data Collection")
    connector = MT5Connector()
    connector.connect()

    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    all_data = {}

    for symbol in symbols:
        print(f"Fetching {symbol}...")
        df = connector.fetch_data(
            symbol,
            mt5.TIMEFRAME_H1,
            datetime.now() - timedelta(days=365*3),
            datetime.now()
        )
        if df is not None and not df.empty:
            all_data[symbol] = df

    connector.disconnect()

    if not all_data:
        print("No data collected. Check MT5 connection.")
        return None, None

    # 2. Feature Engineering
    print("\n[2] Feature Engineering")
    engineer = FeatureEngineer()
    all_features = {}

    for symbol, df in all_data.items():
        print(f"Processing {symbol}...")
        features = engineer.create_features(df)

        # Fit regime model on first symbol, apply to others
        if symbol == list(all_data.keys())[0]:
            engineer.fit_regime_model(features)

        features = engineer.get_regime_specific_features(features)
        all_features[symbol] = features

    # 3. Validation / Training
    print("\n[3] Model Validation")

    # Combine all symbols for training
    combined_features = pd.concat(all_features.values(), keys=all_features.keys())
    combined_features.reset_index(level=0, drop=True, inplace=True)

    # Initialize models
    primary_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )

    meta_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_leaf=50,
        class_weight='balanced_subsample',
        random_state=42
    )

    # Run walk-forward validation
    validator = WalkForwardValidator(primary_model, meta_model)
    results = validator.validate(combined_features)

    # 4. Save results
    print("\n[4] Saving Results")
    results.to_csv('validation_summary.csv')
    print("Results saved to validation_summary.csv")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")

    return validator, engineer


if __name__ == "__main__":
    # Run research mode
    validator, engineer = run_pipeline(mode="research")

"""
Forex-specific feature engineering with regime detection
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.hmm_model = None

    def create_features(self, df, include_target=True):
        """
        Complete feature set for forex trading
        """
        features = pd.DataFrame(index=df.index)

        # ========== PRICE-BASED FEATURES ==========

        # Returns at different horizons
        for horizon in [1, 3, 5, 10, 20]:
            features[f'return_{horizon}'] = df['close'].pct_change(horizon)

        # Log returns (better statistical properties)
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Volatility features
        features['volatility_5'] = features['log_return'].rolling(5).std()
        features['volatility_20'] = features['log_return'].rolling(20).std()
        features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']

        # ========== TREND FEATURES ==========

        # EMAs and ratios
        for period in [10, 20, 50, 200]:
            features[f'ema_{period}'] = ta.ema(df['close'], length=period)
            features[f'price_to_ema_{period}'] = df['close'] / features[f'ema_{period}']

        # Trend strength
        features['trend_strength'] = abs(
            features['ema_50'] - features['ema_200']
        ) / (features['volatility_20'] * np.sqrt(200))

        # ADX for trend intensity
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None and not adx.empty:
            features['adx'] = adx['ADX_14']
        else:
            features['adx'] = 0 # fallback

        # ========== MOMENTUM FEATURES ==========

        # RSI with multiple periods
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = ta.rsi(df['close'], length=period)

        # RSI slope (rate of change)
        if 'rsi_14' in features:
            features['rsi_slope'] = features['rsi_14'].diff(3)

        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            features['macd'] = macd['MACD_12_26_9']
            features['macd_signal'] = macd['MACDs_12_26_9']
            features['macd_hist'] = macd['MACDh_12_26_9']

        # ========== VOLATILITY FEATURES ==========

        # ATR and normalized versions
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        features['atr_14'] = atr
        features['atr_percent'] = features['atr_14'] / df['close']

        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and not bb.empty:
            features['bb_position'] = (df['close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])

        # ========== VOLUME FEATURES (if available) ==========

        if 'tick_volume' in df.columns:
            features['volume'] = df['tick_volume']
            features['volume_ma_20'] = features['volume'].rolling(20).mean()
            features['volume_ratio'] = features['volume'] / features['volume_ma_20']

            # Price-volume relationship
            features['volume_price_trend'] = (
                features['volume'] *
                np.where(df['close'] > df['open'], 1, -1)
            ).rolling(10).sum()

        # ========== TIME FEATURES (Forex seasonality) ==========

        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month

        # Session indicators
        features['london_session'] = (
            (features['hour'] >= 8) & (features['hour'] <= 16)
        ).astype(int)
        features['ny_session'] = (
            (features['hour'] >= 13) & (features['hour'] <= 21)
        ).astype(int)
        features['asia_session'] = (
            (features['hour'] >= 0) & (features['hour'] <= 8)
        ).astype(int)

        # Weekend gap risk (Friday afternoon)
        features['friday_afternoon'] = (
            (features['day_of_week'] == 4) & (features['hour'] >= 18)
        ).astype(int)

        # ========== CROSS-ASSET FEATURES (requires multiple symbols) ==========

        # Placeholder for DXY correlation, gold correlation, etc.
        # These would be added in a multi-symbol pipeline

        # ========== TARGET VARIABLE (for training) ==========

        if include_target:
            # Future returns at different horizons
            for horizon in [1, 3, 5]:
                features[f'target_return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)

                # Binary direction
                features[f'target_direction_{horizon}'] = (
                    features[f'target_return_{horizon}'] > 0
                ).astype(int)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()

        return features

    def fit_regime_model(self, features_df, n_regimes=3):
        """
        Fit Hidden Markov Model for regime detection
        """
        # Use returns and volatility for regime detection
        regime_features = features_df[['log_return', 'volatility_20', 'adx']].dropna()

        self.hmm_model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )

        self.hmm_model.fit(regime_features)
        print(f"HMM converged: {self.hmm_model.monitor_.converged}")

        # Label regimes by characteristics
        means = self.hmm_model.means_
        # Sort by volatility (second column)
        regime_order = np.argsort(means[:, 1])

        self.regime_labels = {
            regime_order[0]: 'low_vol',
            regime_order[1]: 'medium_vol',
            regime_order[2]: 'high_vol'
        }

        return self

    def predict_regime(self, features_df):
        """
        Predict current market regime
        """
        if self.hmm_model is None:
            raise ValueError("Regime model not fitted")

        regime_features = features_df[['log_return', 'volatility_20', 'adx']].dropna()
        hidden_states = self.hmm_model.predict(regime_features)

        # Map to labels
        regime_series = pd.Series(
            [self.regime_labels.get(s, 'unknown') for s in hidden_states],
            index=regime_features.index
        )

        return regime_series

    def get_regime_specific_features(self, features_df):
        """
        Add regime as feature and create regime-specific interactions
        """
        regimes = self.predict_regime(features_df)
        features_df['regime'] = regimes

        # One-hot encode regime
        regime_dummies = pd.get_dummies(regimes, prefix='regime')
        features_df = pd.concat([features_df, regime_dummies], axis=1)

        # Initialize regime columns if they don't exist
        for vol in ['low_vol', 'medium_vol', 'high_vol']:
            col_name = f'regime_{vol}'
            if col_name not in features_df.columns:
                features_df[col_name] = False

        # Regime-volatility interaction
        features_df['regime_vol_interaction'] = (
            features_df['regime_high_vol'].astype(int) * features_df['volatility_20']
        )

        return features_df


# Usage
if __name__ == "__main__":
    from src.data.mt5_connector import MT5Connector
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta

    # Fetch data
    conn = MT5Connector()
    conn.connect()

    df = conn.fetch_data(
        "EURUSD",
        mt5.TIMEFRAME_H1,
        datetime.now() - timedelta(days=365),
        datetime.now()
    )

    if df is not None and not df.empty:
        # Engineer features
        engineer = FeatureEngineer()
        features = engineer.create_features(df)

        print(f"Created {len(features.columns)} features")
        print(features.head())

        # Fit regime model
        engineer.fit_regime_model(features)
        features_with_regime = engineer.get_regime_specific_features(features)

        print("\nRegime distribution:")
        print(features_with_regime['regime'].value_counts())

    conn.disconnect()

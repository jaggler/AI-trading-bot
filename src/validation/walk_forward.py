"""
Purged walk-forward cross-validation for time series
Implements Lopez de Prado's methodology
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class PurgedWalkForwardCV(BaseCrossValidator):
    """
    Walk-forward CV with purging and embargo to prevent leakage
    """
    def __init__(self, n_splits=5, purge_gap=10, embargo_pct=0.01):
        """
        n_splits: Number of walk-forward splits
        purge_gap: Bars to exclude between train and test (prevent leakage)
        embargo_pct: % of test set to exclude at start (reaction to train events)
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate split points
        split_size = n_samples // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            # Train: everything before current split
            train_end = i * split_size

            # Embargo: exclude start of test set
            embargo_size = int(split_size * self.embargo_pct)
            test_start = train_end + self.purge_gap + embargo_size

            # Test: current window
            test_end = min((i + 1) * split_size, n_samples)

            if test_start >= test_end:
                continue

            yield (
                indices[:train_end],
                indices[test_start:test_end]
            )

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class WalkForwardValidator:
    def __init__(self, primary_model, meta_model, cv=None):
        self.primary_model = primary_model
        self.meta_model = meta_model
        self.cv = cv or PurgedWalkForwardCV(n_splits=5)
        self.results = []

    def validate(self, features_df, target_col='target_direction_5'):
        """
        Full walk-forward validation with meta-labeling
        """
        # Drop all target columns except the one we are evaluating
        X = features_df.drop([c for c in features_df.columns if 'target' in c or c == 'regime'], axis=1)
        y = features_df[target_col]

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(self.cv.split(X)):
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{self.cv.get_n_splits()}")
            print(f"{'='*50}")

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Ensure no overlap
            assert X_train.index.max() < X_test.index.min(), "Data leakage detected!"

            # Stage 1: Train primary model
            print("Training primary model...")
            self.primary_model.fit(X_train, y_train)

            # Primary predictions
            primary_train_pred = self.primary_model.predict_proba(X_train)[:, 1]
            primary_test_pred = self.primary_model.predict_proba(X_test)[:, 1]

            # Stage 2: Create meta-labels
            meta_y_train = (primary_train_pred > 0.5) == y_train

            # Stage 3: Train meta-model
            print("Training meta-model...")
            # Meta-features: primary confidence + context features
            meta_X_train = self._create_meta_features(X_train, primary_train_pred)
            meta_X_test = self._create_meta_features(X_test, primary_test_pred)

            self.meta_model.fit(meta_X_train, meta_y_train)

            # Stage 4: Final predictions
            meta_proba = self.meta_model.predict_proba(meta_X_test)[:, 1]

            # Apply threshold
            threshold = 0.6
            final_pred = (meta_proba > threshold).astype(int)

            # Only trade when meta-model agrees with primary
            trade_mask = (primary_test_pred > 0.5) == (meta_proba > 0.5)
            final_pred = final_pred * trade_mask

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, primary_test_pred, meta_proba, final_pred)
            fold_results.append(metrics)

            print(f"Primary Precision: {metrics['primary_precision']:.3f}")
            print(f"Meta Precision: {metrics['meta_precision']:.3f}")
            print(f"Final Precision: {metrics['final_precision']:.3f}")
            print(f"Coverage: {metrics['coverage']:.3f}")

        self.results = fold_results
        return self._summarize_results()

    def _create_meta_features(self, X, primary_confidence):
        """Features for meta-model"""
        meta_features = pd.DataFrame(index=X.index)
        meta_features['primary_conf'] = primary_confidence
        meta_features['primary_pred'] = (primary_confidence > 0.5).astype(int)

        # Context features
        meta_features['volatility_regime'] = pd.qcut(
            X['volatility_20'], q=5, labels=False, duplicates='drop'
        )
        meta_features['trend_strength'] = X['trend_strength']
        meta_features['hour'] = X['hour']

        # Primary model recent performance (would need tracking in live)
        meta_features['rolling_vol'] = X['volatility_20'].rolling(20).mean().values

        return meta_features.fillna(0)

    def _calculate_metrics(self, y_true, primary_proba, meta_proba, final_pred):
        """Calculate comprehensive metrics"""
        primary_pred = (primary_proba > 0.5).astype(int)

        metrics = {
            'primary_precision': precision_score(y_true, primary_pred, zero_division=0),
            'primary_accuracy': accuracy_score(y_true, primary_pred),
            'meta_precision': precision_score(
                (primary_pred == y_true),
                (meta_proba > 0.6),
                zero_division=0
            ),
            'final_precision': precision_score(
                y_true,
                final_pred,
                zero_division=0
            ),
            'final_accuracy': accuracy_score(y_true, final_pred),
            'coverage': (final_pred != 0).mean(),
            'meta_threshold': 0.6
        }

        return metrics

    def _summarize_results(self):
        """Aggregate results across folds"""
        summary = pd.DataFrame(self.results)

        print(f"\n{'='*50}")
        print("WALK-FORWARD VALIDATION SUMMARY")
        print(f"{'='*50}")

        for col in summary.columns:
            if col != 'meta_threshold':
                print(f"{col:20s}: {summary[col].mean():.3f} (+/- {summary[col].std():.3f})")

        return summary

    def plot_results(self):
        """Visualize validation results"""
        if not self.results:
            print("No results to plot. Run validate() first.")
            return

        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Precision comparison
        ax = axes[0, 0]
        df[['primary_precision', 'meta_precision', 'final_precision']].plot(kind='bar', ax=ax)
        ax.set_title('Precision Comparison Across Folds')
        ax.set_ylabel('Precision')
        ax.legend(['Primary', 'Meta', 'Final'])

        # Coverage vs Precision trade-off
        ax = axes[0, 1]
        ax.scatter(df['coverage'], df['final_precision'])
        ax.set_xlabel('Coverage (fraction of trades taken)')
        ax.set_ylabel('Final Precision')
        ax.set_title('Coverage vs Precision Trade-off')

        # Metrics over time
        ax = axes[1, 0]
        df[['primary_precision', 'final_precision']].plot(ax=ax)
        ax.set_title('Precision Over Folds (Time)')
        ax.set_ylabel('Precision')

        # Distribution of predictions
        ax = axes[1, 1]
        df['coverage'].hist(ax=ax, bins=10)
        ax.set_title('Distribution of Trade Coverage')
        ax.set_xlabel('Coverage')

        plt.tight_layout()
        plt.savefig('validation_results.png')
        plt.show()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def create_pipeline():
    """Create a basic Scikit-learn pipeline for the model."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
    return pipeline

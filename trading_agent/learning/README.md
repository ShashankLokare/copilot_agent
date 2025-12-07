# ML Prediction Engine Documentation

## Overview

The ML Prediction Engine is a comprehensive framework for building, training, evaluating, and deploying machine learning models for stock price forecasting. It handles the complete pipeline from feature engineering to signal generation for live trading.

## Architecture

```
Data Layer
  ├── OHLCV Price Data
  ├── Technical Indicators
  └── Market Regime

Feature Layer
  ├── Feature Engineering (learning/features.py)
  └── Feature Vectors (learning/data_structures.py)

Training Layer
  ├── Label Generation (features/labels.py)
  ├── Walk-Forward Splitter (learning/walk_forward.py)
  └── Data Preparation (learning/walk_forward.py)

Model Layer
  ├── Direction Classifier (XGBoost/LightGBM)
  ├── Return Regressor (XGBoost/LightGBM)
  └── Probability Calibration (Isotonic)

Evaluation Layer
  ├── Classification Metrics (accuracy, precision, recall, ROC-AUC)
  ├── Regression Metrics (MAE, RMSE, R²)
  ├── Trading Metrics (hit rates, conditional returns)
  └── Per-Regime Analysis (learning/performance_analyzer.py)

Deployment Layer
  ├── Signal Generation (alpha/ml_alpha_xgboost.py)
  ├── Model Persistence (learning/model_store.py)
  └── Configuration Management (config/ml_model_config.py)

Backtest Integration
  └── ALPHATrader with MLAlpha signals
```

## Core Modules

### 1. Data Structures (learning/data_structures.py)

Defines strongly-typed interfaces for the entire pipeline:

```python
from learning.data_structures import (
    FeatureVector,      # Input: features for a single bar
    Label,              # Output: k-bar forward return + direction
    PredictionOutput,   # Model output: probabilities + expected return
    TrainingDataset,    # Aligned feature-label pairs
    ModelMetadata,      # Versioning + hyperparameters
    EvaluationMetrics,  # Classification + regression + trading metrics
)
```

**FeatureVector**: Represents features for prediction
```python
fv = FeatureVector(
    timestamp=pd.Timestamp("2024-01-15"),
    symbol="AAPL",
    features={"sma_ratio": 1.02, "volatility": 0.02, "momentum": 0.001},
    regime="uptrend",
)
```

**Label**: Forward-looking classification + regression targets
```python
label = Label(
    timestamp=pd.Timestamp("2024-01-15"),
    symbol="AAPL",
    horizon_bars=20,  # 20-bar forward prediction
    target_return_k=0.02,  # 2% forward return
    target_direction_k=1,  # Direction: -1 (down), 0 (neutral), +1 (up)
)
```

**PredictionOutput**: What the model produces
```python
pred = PredictionOutput(
    timestamp=pd.Timestamp("2024-01-15"),
    symbol="AAPL",
    prob_up=0.65,  # 65% probability of upward move
    prob_down=0.35,
    expected_return=0.015,  # 1.5% expected return
    confidence=0.65,  # max(prob_up, prob_down)
)
```

### 2. Label Generation (features/labels.py)

Generates forward-looking targets from price data without look-ahead bias:

```python
from features.labels import LabelGenerator

label_gen = LabelGenerator(
    horizons=[5, 20],  # Predict 5-bar and 20-bar returns
    direction_threshold_pct=0.5,  # ±0.5% threshold for classification
    neutral_band_pct=0.1,  # ±0.1% neutral zone
)

# Generate labels from OHLCV data
labels_df = label_gen.generate_labels_from_ohlcv(ohlcv_df)

# Split by horizon
labels_by_horizon = label_gen.split_by_horizon(labels_df)
labels_5bar = labels_by_horizon[5]
labels_20bar = labels_by_horizon[20]

# Split by regime
labels_by_regime = label_gen.split_by_regime(labels_df, regime_df)
uptrend_labels = labels_by_regime["uptrend"]
```

**Key Safety Features**:
- **No look-ahead bias**: Uses price[t+k]/price[t] - 1 for forward returns
- **Configurable thresholds**: Customize what counts as "up" vs "down"
- **Neutral band**: Avoids over-fitting to small moves
- **Multi-horizon**: Train separate models for different prediction periods

### 3. Walk-Forward Validation (learning/walk_forward.py)

Time-based cross-validation with strict anti-leakage:

```python
from learning.walk_forward import WalkForwardSplitter, DataPreparer

# Create walk-forward splitter
splitter = WalkForwardSplitter(
    train_window_days=504,  # ~2 years training
    valid_window_days=252,  # ~1 year validation
    step_days=63,  # Quarterly rolling
    gap_days=5,  # 5-day gap to prevent leakage
)

# Generate folds
folds = splitter.generate_folds(df, min_train_samples=100)

for fold in folds:
    # fold.train_start, fold.train_end, fold.valid_start, fold.valid_end
    train_data = df[df.timestamp >= fold.train_start & df.timestamp <= fold.train_end]
    valid_data = df[df.timestamp >= fold.valid_start & df.timestamp <= fold.valid_end]
    # Train model, evaluate on valid_data
```

**Anti-Leakage Guarantees**:
- Validation always comes after training
- Optional gap between train and valid periods
- Forward-only time progression (no future data in training)

### 4. Prediction Engine (learning/prediction_engine.py)

Trains and deploys ML models for price prediction:

```python
from learning.prediction_engine import PredictionEngine
import pandas as pd

# Create engine with config
config = {
    "model_type": "xgboost",
    "regime_mode": "single_model_with_feature",
    "xgboost_params": {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
    }
}

engine = PredictionEngine(config=config)

# Train on labeled data
engine.fit(
    train_df=train_df,
    valid_df=valid_df,
    feature_columns=["sma_ratio", "volatility", "momentum"],
    targets={
        "target_return_5": "target_return_5",
        "target_return_20": "target_return_20",
        "target_direction_5": "target_direction_5",
        "target_direction_20": "target_direction_20",
    }
)

# Make predictions
predictions = engine.predict(test_df)
# Returns list of PredictionOutput objects

# Save model
engine.save("models/my_model")

# Load model
engine = PredictionEngine.load("models/my_model")
```

**Features**:
- Trains separate classifiers for direction + regressors for returns
- Supports XGBoost and LightGBM
- Isotonic probability calibration
- Per-horizon models (independent for each forward period)
- Feature scaling fitted only on training data (no leakage)

### 5. Performance Analysis (learning/performance_analyzer.py)

Comprehensive metrics computation and reliability checking:

```python
from learning.performance_analyzer import PerformanceAnalyzer, ReliabilityChecker

# Compute classification metrics
class_metrics = PerformanceAnalyzer.compute_classification_metrics(
    y_true, y_pred, y_pred_proba=probs
)
# Returns: {accuracy, precision, recall, f1, roc_auc, brier_score}

# Compute trading-specific metrics
trading_metrics = PerformanceAnalyzer.compute_trading_metrics(
    y_true_return, y_pred_proba, y_pred_return,
    prob_thresholds=[0.55, 0.60, 0.65, 0.70]
)
# Returns: {hit_rate_55, hit_rate_60, ..., avg_return_if_signal, return_correlation}

# Evaluate on validation fold
metrics = PerformanceAnalyzer.evaluate_on_fold(
    y_true_ret, y_true_dir, y_pred_proba, y_pred_ret, regimes=regime_array
)

# Check if model is deployment-ready
is_deployable, warnings = ReliabilityChecker.check_deployment_criteria(metrics)

# Print formatted report
ReliabilityChecker.print_report(metrics, fold_id=0)
```

**Metrics Computed**:
- **Classification**: Accuracy, Precision, Recall, F1, ROC-AUC, Brier Score
- **Regression**: MAE, RMSE, R²
- **Trading**: Hit rates at various confidence levels, conditional returns
- **Per-Regime**: All metrics broken down by market regime

### 6. ML Alpha Integration (alpha/ml_alpha_xgboost.py)

Converts model predictions to trading signals:

```python
from alpha.ml_alpha_xgboost import MLAlpha, MLAlphaConfig
from learning.model_store import load_model

# Load trained model
model = load_model("xgboost_model")

# Configure alpha
config = MLAlphaConfig(
    long_prob_threshold=0.60,
    short_prob_threshold=0.40,
    position_size_mode="prob_weighted",  # Size by confidence
    max_position_size=1.0,
    holding_bars=20,
)

# Create alpha
alpha = MLAlpha(config=config, model=model)

# Generate signal for single row
signal = alpha.generate_signal(
    timestamp=pd.Timestamp("2024-01-15"),
    symbol="AAPL",
    features={"sma_ratio": 1.02, "volatility": 0.02},
    regime="uptrend",
)

# Or batch generate from DataFrame
signals = alpha.generate_signals_batch(
    df=prediction_df,
    feature_columns=["sma_ratio", "volatility"],
    regime_column="regime",
)
```

### 7. Model Persistence (learning/model_store.py)

Save, load, and version trained models:

```python
from learning.model_store import ModelStore, save_model, load_model

# Save trained model
store = ModelStore(base_path="models")
path = store.save_model(
    model=trained_engine,
    model_name="xgboost_5bar_20bar",
    version="v1.0",
    tags={"accuracy": 0.53, "dataset": "2023_2024"},
)

# Load latest version
model = load_model("xgboost_5bar_20bar")

# Load specific version
model = load_model("xgboost_5bar_20bar", version="2024-01-15")

# List all models and versions
models = store.list_models()
# {"xgboost_5bar_20bar": ["v1.0", "v1.1"], "lightgbm_model": ["v0.1"]}

# Get metadata without loading full model
metadata = store.get_metadata("xgboost_5bar_20bar")
```

## End-to-End Workflow

### 1. Load and Prepare Data

```python
import pandas as pd

# Load OHLCV data
df = pd.read_csv("data/ohlcv.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp")
```

### 2. Engineer Features

```python
# Calculate technical indicators
df['sma_5'] = df['close'].rolling(5).mean()
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_ratio'] = df['close'] / df['sma_20']
df['volatility'] = df['close'].pct_change().rolling(20).std()
df['momentum'] = df['close'].pct_change().rolling(10).mean()
# ... more features
```

### 3. Generate Labels

```python
from features.labels import LabelGenerator

label_gen = LabelGenerator(horizons=[5, 20])
labels_df = label_gen.generate_labels_from_ohlcv(df)

# Merge features and labels
training_df = df.merge(
    labels_df,
    on=['timestamp', 'symbol'],
    how='inner'
)
```

### 4. Split Data (Walk-Forward)

```python
from learning.walk_forward import WalkForwardSplitter

splitter = WalkForwardSplitter(
    train_window_days=504,
    valid_window_days=252,
    step_days=63,
    gap_days=5,
)

folds = splitter.generate_folds(training_df)
```

### 5. Train Models

```python
from learning.prediction_engine import PredictionEngine

for fold in folds:
    train_df = training_df[training_df.timestamp.between(fold.train_start, fold.train_end)]
    valid_df = training_df[training_df.timestamp.between(fold.valid_start, fold.valid_end)]
    
    model = PredictionEngine(config={
        "model_type": "xgboost",
        "xgboost_params": {"max_depth": 6, "learning_rate": 0.1}
    })
    
    model.fit(
        train_df=train_df,
        valid_df=valid_df,
        feature_columns=["sma_ratio", "volatility", "momentum"],
        targets={
            "target_return_5": "target_return_5",
            "target_return_20": "target_return_20",
            "target_direction_5": "target_direction_5",
            "target_direction_20": "target_direction_20",
        }
    )
    
    # Evaluate
    predictions = model.predict(valid_df)
```

### 6. Evaluate Performance

```python
from learning.performance_analyzer import (
    PerformanceAnalyzer,
    ReliabilityChecker,
)

metrics = PerformanceAnalyzer.evaluate_on_fold(
    y_true_ret, y_true_dir, y_pred_proba, y_pred_ret
)

is_ok, warnings = ReliabilityChecker.check_deployment_criteria(metrics)
ReliabilityChecker.print_report(metrics)
```

### 7. Deploy Model

```python
from learning.model_store import save_model, load_model
from alpha.ml_alpha_xgboost import MLAlpha, MLAlphaConfig

# Save best model
save_model(best_model, "xgboost_production")

# Load and deploy
model = load_model("xgboost_production")
alpha = MLAlpha(
    config=MLAlphaConfig(long_prob_threshold=0.60),
    model=model
)

# Use in trading system
signals = alpha.generate_signals_batch(live_df, feature_columns=features)
```

## Configuration

All hyperparameters are centralized in `config/ml_model_config.py`:

```python
from config.ml_model_config import (
    get_xgboost_params,
    get_label_config,
    get_walk_forward_config,
    get_alpha_config,
)

# Get configuration sections
xgb_params = get_xgboost_params()
label_cfg = get_label_config()
wf_cfg = get_walk_forward_config()
alpha_cfg = get_alpha_config()
```

## Performance Metrics

The system tracks three categories of metrics:

### Classification Metrics (Direction Prediction)
- **Accuracy**: % of correct up/down predictions
- **Precision/Recall**: True positive rates
- **F1 Score**: Harmonic mean of precision/recall
- **ROC-AUC**: Area under ROC curve
- **Brier Score**: Probability calibration (lower is better)

### Regression Metrics (Return Prediction)
- **MAE**: Mean absolute error in return prediction
- **RMSE**: Root mean squared error
- **R² Score**: Variance explained by model

### Trading Metrics
- **Hit Rate @ X%**: % correct when prob > threshold
- **Avg Return if Signal**: Average return when taking a signal
- **Return Correlation**: Correlation between predicted and actual returns
- **Per-Regime Analysis**: All metrics broken down by market regime

## Best Practices

1. **Always use walk-forward validation** for time-series data (prevents look-ahead bias)
2. **Check per-regime metrics** to ensure model generalizes across market conditions
3. **Monitor Brier score** for probability calibration (use isotonic calibration)
4. **Use high confidence thresholds** for live trading (60-70% recommended)
5. **Retrain regularly** (monthly or quarterly) with new data
6. **Keep training/validation/test periods separate** (no overlap)
7. **Log all hyperparameters** and model metadata for reproducibility
8. **Start conservative** with position sizing and gradually increase if metrics hold

## Troubleshooting

**Problem**: Model accuracy < 50% (worse than random)
- **Solution**: Check for data leakage (validation data in training features)
- **Check**: Are you using future information in features?
- **Try**: Different horizons, feature engineering, or market regimes

**Problem**: High variance in per-regime performance
- **Solution**: Train separate models per regime
- **Try**: Increasing `regime_mode` to "per_regime_models"
- **Alternative**: Filter signals to only high-confidence regimes

**Problem**: Poor probability calibration (high Brier score)
- **Solution**: Enable isotonic calibration (already enabled by default)
- **Try**: Different model type (XGBoost vs LightGBM)

**Problem**: Models underperform on new data
- **Solution**: Retrain more frequently
- **Try**: Shorter walk-forward windows
- **Alternative**: Use ensemble of recent models

## Integration with ALPHATrader

The ML prediction engine integrates seamlessly with the trading system:

```python
from trading.trader import ALPHATrader
from alpha.ml_alpha_xgboost import MLAlpha
from learning.model_store import load_model

# Load trained model
model = load_model("xgboost_production")

# Create alpha
alpha = MLAlpha(config=MLAlphaConfig(), model=model)

# Initialize trader with ML alpha
trader = ALPHATrader(alphas=[alpha, ...])

# Run backtest or live trading
trader.backtest(...)
# or
trader.live(...)
```

## References

- **Label Generation**: `features/labels.py`
- **Data Structures**: `learning/data_structures.py`
- **Prediction Engine**: `learning/prediction_engine.py`
- **Walk-Forward Validation**: `learning/walk_forward.py`
- **Performance Analysis**: `learning/performance_analyzer.py`
- **ML Alpha**: `alpha/ml_alpha_xgboost.py`
- **Model Storage**: `learning/model_store.py`
- **Example Training**: `learning/train_model_example.py`
- **Configuration**: `config/ml_model_config.py`

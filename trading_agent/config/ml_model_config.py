"""
ML Model Configuration

Central configuration for training, model selection, hyperparameters, and deployment.
"""

ml_config = {
    # Model type selection
    "model": {
        "type": "xgboost",  # "xgboost" or "lightgbm"
        "regime_mode": "single_model_with_feature",  # "single_model_with_feature" or "per_regime_models"
    },
    
    # Target horizons (bars into the future to predict)
    "horizons": [5, 20],
    
    # Label generation
    "labels": {
        "direction_threshold_pct": 0.5,  # ±0.5% threshold for direction classification
        "neutral_band_pct": 0.1,  # ±0.1% neutral zone around zero return
        "balance_classes": False,  # Whether to undersample majority class
    },
    
    # Walk-forward training configuration
    "walk_forward": {
        "train_window_days": 504,  # ~2 years
        "valid_window_days": 252,  # ~1 year
        "step_days": 63,  # Quarterly stepping
        "gap_days": 5,  # Gap between train and validation to prevent leakage
    },
    
    # XGBoost hyperparameters
    "xgboost": {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0,  # L1 regularization
        "reg_lambda": 1.0,  # L2 regularization
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds": 20,
    },
    
    # LightGBM hyperparameters (alternative to XGBoost)
    "lightgbm": {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    },
    
    # Feature scaling
    "scaling": {
        "method": "standard",  # "standard" or "robust"
        "per_horizon": True,  # Whether to fit separate scalers for each horizon
    },
    
    # Probability calibration
    "calibration": {
        "method": "isotonic",  # "isotonic", "sigmoid", or None
        "apply_to_classifier": True,
    },
    
    # Performance thresholds for deployment
    "deployment": {
        "min_accuracy": 0.50,
        "min_hit_rate_60": 0.52,  # Better than 50% random
        "max_brier_score": 0.25,  # Probability calibration
        "max_regime_variance": 0.10,  # Consistency across regimes
    },
    
    # Feature engineering defaults
    "features": {
        "default_features": [
            "sma_5",
            "sma_20",
            "sma_ratio",
            "volatility",
            "momentum",
            "volume_ratio",
        ],
        "scaling": "standard",
    },
    
    # ML Alpha generation
    "ml_alpha": {
        "long_prob_threshold": 0.60,
        "short_prob_threshold": 0.40,
        "position_size_mode": "fixed",  # "fixed" or "prob_weighted"
        "max_position_size": 1.0,
        "holding_bars": 20,
        "min_confidence": 0.5,
        "min_expected_return": 0.0001,
        "filter_by_regime": False,
        "allowed_regimes": None,  # e.g., ["uptrend", "strong_uptrend"]
    },
    
    # Model storage
    "storage": {
        "base_path": "models",
        "keep_best_n_versions": 5,
        "archive_old_versions": True,
    },
    
    # Training data sources
    "data": {
        "source": "database",  # "database", "csv", "api"
        "database": {
            "connection_string": "sqlite:///data/market_data.db",
            "symbols": None,  # None = all symbols in database
        },
        "csv": {
            "path": "data/ohlcv.csv",
            "date_column": "timestamp",
            "symbol_column": "symbol",
        },
    },
    
    # Logging
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/ml_training.log",
    },
    
    # Random state for reproducibility
    "random_state": 42,
}


def get_xgboost_params():
    """Get XGBoost hyperparameters."""
    return ml_config["xgboost"]


def get_lightgbm_params():
    """Get LightGBM hyperparameters."""
    return ml_config["lightgbm"]


def get_model_config():
    """Get model selection config."""
    return ml_config["model"]


def get_label_config():
    """Get label generation config."""
    return ml_config["labels"]


def get_walk_forward_config():
    """Get walk-forward training config."""
    return ml_config["walk_forward"]


def get_alpha_config():
    """Get ML alpha generation config."""
    return ml_config["ml_alpha"]


def get_deployment_thresholds():
    """Get deployment performance thresholds."""
    return ml_config["deployment"]


if __name__ == "__main__":
    import json
    
    # Print configuration in readable format
    print("ML Configuration:")
    print("=" * 70)
    print(json.dumps(ml_config, indent=2, default=str))

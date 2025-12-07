#!/usr/bin/env python3
"""
NIFTY50 Advanced ML Training & Robustness Workflow

Upgrades:
- Config-driven XGBoost/LightGBM models with class-imbalance handling
- Walk-forward cross-validation with isotonic calibration
- Realistic NSE trading costs baked into signal PnL
- Monte Carlo block-bootstrap robustness test on strategy returns
- Model registry + model store integration

Quick start:
    python scripts/train_nifty50_models.py --symbols 50 --train-years 3 --test-years 1
"""

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - handled at runtime
    xgb = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - handled at runtime
    lgb = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ml_training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Project imports
from scripts.collect_nifty50_15y_data import collect_nifty50_data
from scripts.prepare_nifty50_data import NiftyDataPreparation, prepare_nifty50_data
from learning.performance_analyzer import PerformanceAnalyzer
from learning.model_store import ModelStore
from learning.model_metadata import ModelMetadata, ModelRegistry
from backtest.strategy_grader import StrategyGrader
from config.ml_model_config import (
    get_lightgbm_params,
    get_walk_forward_config,
    get_xgboost_params,
)


def _build_model(model_type: str, imbalance: float, pos_weight_mult: float = 1.0):
    """Factory for gradient boosting models with imbalance handling."""
    if model_type == "xgboost":
        if xgb is None:
            raise ImportError("XGBoost not installed. Run bash install_ml_dependencies.sh")
        params = copy.deepcopy(get_xgboost_params())
        params["eval_metric"] = "logloss"
        # Avoid early_stopping_rounds requirement inside the constructor; handle validation externally
        params.pop("early_stopping_rounds", None)
        params["n_jobs"] = params.get("n_jobs", 4)
        params["scale_pos_weight"] = imbalance * pos_weight_mult
        return xgb.XGBClassifier(**params), params
    if model_type == "lightgbm":
        if lgb is None:
            raise ImportError("LightGBM not installed. Run bash install_ml_dependencies.sh")
        params = copy.deepcopy(get_lightgbm_params())
        params["n_jobs"] = params.get("n_jobs", 4)
        params["class_weight"] = {0: 1.0, 1: imbalance * pos_weight_mult}
        return lgb.LGBMClassifier(**params), params
    raise ValueError(f"Unknown model type: {model_type}")


def _load_existing_dataset(path: Path) -> pd.DataFrame:
    """Load an existing OHLCV dataset and normalize columns."""
    df = pd.read_csv(path)
    # Normalize column casing
    df.columns = [c.lower() for c in df.columns]

    # Allow alternate date column name
    if "timestamp" not in df.columns and "date" in df.columns:
        df["timestamp"] = df["date"]

    if "timestamp" not in df.columns:
        raise ValueError(f"Dataset at {path} missing 'timestamp' column")

    # Enforce required columns
    required = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset at {path} missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    df = df[required]
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def _get_current_nifty50() -> List[str]:
    """Return current NIFTY50 symbol list (kept in collect script)."""
    try:
        from scripts.collect_nifty50_15y_data import NIFTY50_SYMBOLS
        return list(NIFTY50_SYMBOLS)
    except Exception:
        return []


def _build_trades_df(
    closes: np.ndarray,
    timestamps: np.ndarray,
    positions: np.ndarray,
    cost_decimal: float,
) -> pd.DataFrame:
    """Create minimal trades DataFrame for strategy grading."""
    trades = []
    for i, pos in enumerate(positions[:-1]):  # last bar has no exit
        if pos == 0:
            continue
        entry_price = closes[i] * (1 + cost_decimal if pos == 1 else 1 - cost_decimal)
        exit_price = closes[i + 1] * (1 - cost_decimal if pos == 1 else 1 + cost_decimal)
        trades.append(
            {
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": int(pos),
                "entry_date": timestamps[i],
                "exit_date": timestamps[i + 1],
            }
        )
    return pd.DataFrame(trades)


def run_complete_training_pipeline(
    num_symbols: int = 50,
    train_years: int = 3,
    test_years: int = 1,
    model_type: str = "xgboost",
    max_folds: int = 5,
    prob_long: float = 0.52,
    prob_short: float = 0.48,
    pos_weight_mult: float = 1.0,
    n_monte_carlo: int = 500,
    use_existing_data: bool = True,
    raw_data_path: str = "data/nifty50_15y_ohlcv.csv",
    use_current_nifty50: bool = False,
) -> Dict:
    """
    Run end-to-end training, calibration, realistic backtest, and Monte Carlo.
    """
    logger.info("=" * 70)
    logger.info("NIFTY50 ADVANCED ML TRAINING PIPELINE")
    logger.info("=" * 70)

    results = {
        "status": "started",
        "timestamp": pd.Timestamp.now().isoformat(),
        "stages": {},
    }

    # Stage 1: Collect Data
    logger.info("\n[STAGE 1] COLLECTING 15 YEARS OF NIFTY50 DATA")
    logger.info("-" * 70)
    data_path = Path("data/nifty50_15y_ohlcv.csv")
    df_raw = None
    if raw_data_path:
        data_path = Path(raw_data_path)

    if use_existing_data and data_path.exists():
        try:
            df_raw = _load_existing_dataset(data_path)
            # Persist normalized schema (adds timestamp, standard columns) for downstream steps
            df_raw.to_csv(data_path, index=False)
            results["stages"]["data_collection"] = {
                "status": "success",
                "rows": len(df_raw),
                "symbols": df_raw["symbol"].nunique(),
                "date_range": f"{df_raw['timestamp'].min().date()} to {df_raw['timestamp'].max().date()} (existing file)",
                "source": "existing_file",
            }
            logger.info(f"✓ Using existing data file at {data_path}")
        except Exception as e:
            logger.warning(f"Failed to load existing data file, will attempt collection: {e}")
            df_raw = None

    if df_raw is None:
        try:
            df_raw, _ = collect_nifty50_data(
                output_path=str(data_path),
                start_year=2010,
                end_year=2025,
                use_synthetic_fallback=True,
            )
            results["stages"]["data_collection"] = {
                "status": "success",
                "rows": len(df_raw),
                "symbols": df_raw["symbol"].nunique(),
                "date_range": f"{df_raw['timestamp'].min().date()} to {df_raw['timestamp'].max().date()}",
                "source": "collected",
            }
            logger.info("✓ Data collection complete")
        except Exception as e:  # pragma: no cover - handled in runtime flow
            logger.error(f"✗ Data collection failed: {e}")
            results["stages"]["data_collection"] = {"status": "failed", "error": str(e)}
            return results
    # Optional symbol filter
    if use_current_nifty50:
        allow = set(_get_current_nifty50())
        before = df_raw["symbol"].nunique()
        df_raw = df_raw[df_raw["symbol"].isin(allow)].copy()
        after = df_raw["symbol"].nunique()
        results["stages"]["data_collection"]["symbols_filtered"] = f"{before} -> {after}"
        logger.info(f"Filtered symbols to current NIFTY50: {after} kept (from {before})")
        # Persist filtered version for downstream prep
        df_raw.to_csv(data_path, index=False)

    # Stage 2: Prepare Data
    logger.info("\n[STAGE 2] PREPARING DATA (FEATURES + LABELS)")
    logger.info("-" * 70)
    try:
        df_training, feature_cols = prepare_nifty50_data(
            raw_data_path=str(data_path),
            output_path="data/nifty50_training_data.csv",
        )
        # Add forward return for strategy evaluation
        # Use per-symbol next-bar close to compute forward return without misaligned shapes
        next_close = df_training.groupby("symbol")["close"].shift(-1)
        df_training["forward_return"] = (next_close / df_training["close"]) - 1
        df_training = df_training.dropna(subset=["forward_return"])

        results["stages"]["data_preparation"] = {
            "status": "success",
            "rows": len(df_training),
            "features": len(feature_cols),
            "symbols": df_training["symbol"].nunique(),
        }
        logger.info("✓ Data preparation complete")
    except Exception as e:
        logger.error(f"✗ Data preparation failed: {e}")
        results["stages"]["data_preparation"] = {"status": "failed", "error": str(e)}
        return results

    # Stage 3: Walk-Forward Splits
    logger.info("\n[STAGE 3] CREATING WALK-FORWARD SPLITS")
    logger.info("-" * 70)
    try:
        prep = NiftyDataPreparation(raw_data_path=str(data_path))
        wf_cfg = get_walk_forward_config()
        splits = prep.get_walk_forward_splits(
            df=df_training,
            train_window_days=wf_cfg.get("train_window_days", 252 * train_years),
            test_window_days=wf_cfg.get("valid_window_days", 252 * test_years),
            step_size_days=wf_cfg.get("step_days", 252),
        )[:max_folds]
        results["stages"]["walk_forward_splits"] = {
            "status": "success",
            "num_splits": len(splits),
        }
        logger.info(f"✓ Created {len(splits)} walk-forward splits (capped at {max_folds})")
    except Exception as e:
        logger.error(f"✗ Walk-forward split creation failed: {e}")
        results["stages"]["walk_forward_splits"] = {"status": "failed", "error": str(e)}
        return results

    # Stage 4: Train + Calibrate + Backtest
    logger.info("\n[STAGE 4] TRAINING MODELS WITH CALIBRATION & COSTED BACKTEST")
    logger.info("-" * 70)
    try:
        cost_cfg = StrategyGrader().config.get("costs", {})
        cost_bps = (
            cost_cfg.get("slippage_bps", 0.0)
            + cost_cfg.get("commission_bps", 0.0)
            + cost_cfg.get("bid_ask_spread_bps", 0.0)
        )
        cost_decimal = cost_bps / 10000.0

        trained_models = []
        fold_results: List[Dict] = []
        all_returns: List[float] = []

        for fold_idx, (train_df, test_df) in enumerate(splits):
            logger.info(f"\nFold {fold_idx + 1}/{len(splits)}")
            train_df = train_df.dropna(subset=["forward_return"])
            test_df = test_df.dropna(subset=["forward_return"])

            # Keep feature names intact for downstream models (LightGBM warning avoidance)
            X_train = train_df[feature_cols]
            y_train = train_df["label"].values
            X_test = test_df[feature_cols]
            y_test = test_df["label"].values

            pos = np.sum(y_train == 1)
            neg = np.sum(y_train == 0)
            imbalance = (neg / (pos + 1e-6)) if pos > 0 else 1.0

            base_model, model_params = _build_model(model_type, imbalance, pos_weight_mult)

            # Calibration split (time-respecting)
            X_subtrain, X_calib, y_subtrain, y_calib = train_test_split(
                X_train, y_train, test_size=0.2, shuffle=False
            )
            if model_type == "lightgbm":
                base_model.fit(X_subtrain, y_subtrain)
            else:
                base_model.fit(X_subtrain, y_subtrain, eval_set=[(X_calib, y_calib)], verbose=False)
            calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
            calibrated_model.fit(X_train, y_train)

            pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
            pred_class = (pred_proba > 0.5).astype(int)

            class_metrics = PerformanceAnalyzer.compute_classification_metrics(
                y_test,
                pred_class,
                pred_proba,
            )

            # Cost-aware strategy returns
            positions = np.zeros_like(pred_proba)
            positions[pred_proba > prob_long] = 1
            positions[pred_proba < prob_short] = -1
            trade_mask = positions != 0
            strategy_returns = positions * test_df["forward_return"].values - cost_decimal * trade_mask
            all_returns.extend(strategy_returns.tolist())

            trades_df = _build_trades_df(
                test_df["close"].values,
                test_df["timestamp"].values,
                positions,
                cost_decimal,
            )

            fold_results.append(
                {
                    "fold": fold_idx,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "metrics": class_metrics,
                    "prob_thresholds": {"long": prob_long, "short": prob_short},
                    "avg_signal_rate": float(np.mean(trade_mask)),
                    "returns": strategy_returns.tolist(),
                    "trades": trades_df,
                    "model": calibrated_model,
                    "test_auc": class_metrics.get("roc_auc", 0.0),
                    "test_accuracy": class_metrics.get("accuracy", 0.0),
                    "cost_bps": cost_bps,
                }
            )

            trained_models.append(calibrated_model)
            logger.info(
                "  AUC: %.4f | Acc: %.4f | Signals: %.1f%% days"
                % (
                    class_metrics.get("roc_auc", 0),
                    class_metrics.get("accuracy", 0),
                    np.mean(trade_mask) * 100,
                )
            )

        best_fold = max(enumerate(fold_results), key=lambda x: x[1]["test_auc"])
        best_model = trained_models[best_fold[0]]

        results["stages"]["model_training"] = {
            "status": "success",
            "num_folds": len(fold_results),
            "best_fold": best_fold[0],
            "best_test_auc": best_fold[1]["test_auc"],
            "best_test_accuracy": best_fold[1]["test_accuracy"],
            "fold_results": fold_results,
        }
        results["all_returns"] = all_returns
        results["best_model"] = best_model
        logger.info(
            f"\n✓ Model training complete (best fold: {best_fold[0]}, AUC: {best_fold[1]['test_auc']:.4f})"
        )
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        results["stages"]["model_training"] = {"status": "failed", "error": str(e)}
        return results

    # Stage 5: Monte Carlo Robustness
    logger.info("\n[STAGE 5] MONTE CARLO ROBUSTNESS (BLOCK BOOTSTRAP)")
    logger.info("-" * 70)
    try:
        strat_grader = StrategyGrader()
        mc_results = strat_grader.monte_carlo_analysis(
            np.array(all_returns),
            n_simulations=n_monte_carlo,
            confidence_level=strat_grader.config.get("monte_carlo", {}).get("confidence_level", 0.95),
        )
        results["stages"]["monte_carlo"] = {
            "status": "success",
            "prob_profitable": mc_results.get("prob_profitable", None),
            "sharpe_ci": (
                mc_results.get("sharpe_ci_lower"),
                mc_results.get("sharpe_ci_upper"),
            ),
            "return_ci": (
                mc_results.get("return_ci_lower"),
                mc_results.get("return_ci_upper"),
            ),
        }
        logger.info(
            f"✓ MC Sharpe CI: [{mc_results.get('sharpe_ci_lower'):.2f}, "
            f"{mc_results.get('sharpe_ci_upper'):.2f}] | "
            f"Prob Profitable Paths: {mc_results.get('prob_profitable', 0):.2%}"
        )
    except Exception as e:
        logger.error(f"✗ Monte Carlo analysis failed: {e}")
        results["stages"]["monte_carlo"] = {"status": "failed", "error": str(e)}

    # Stage 6: Register + Persist
    logger.info("\n[STAGE 6] REGISTERING MODEL IN REGISTRY")
    logger.info("-" * 70)
    try:
        model_store = ModelStore()
        registry = ModelRegistry()

        best_model_id = f"nifty50_{model_type}_adv_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = model_store.save_model(results["best_model"], best_model_id)

        metadata = ModelMetadata(
            model_name=best_model_id,
            model_version="1.0",
            model_type=model_type,
            training_date=pd.Timestamp.now().isoformat(),
            training_features=feature_cols,
            training_horizons=[1],
            training_data_period={
                "start": str(df_training["timestamp"].min().date()),
                "end": str(df_training["timestamp"].max().date()),
            },
            training_n_samples=len(df_training),
            hyperparameters=model_params,
            deployment_status="INCUBATION",
        )
        registry.register_model(metadata)

        results["stages"]["model_registration"] = {
            "status": "success",
            "model_id": best_model_id,
            "model_path": str(model_path),
            "deployment_status": "INCUBATION",
        }
        results["status"] = "completed"
        results["best_model_id"] = best_model_id
    except Exception as e:
        logger.error(f"✗ Model registration failed: {e}")
        results["stages"]["model_registration"] = {"status": "failed", "error": str(e)}

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Data: {len(df_training):,} bars × {len(feature_cols)} features")
    logger.info(f"Symbols: {df_training['symbol'].nunique()}")
    logger.info(
        f"Date Range: {df_training['timestamp'].min().date()} to {df_training['timestamp'].max().date()}"
    )
    logger.info(f"Walk-Forward Splits: {len(splits)} | Best Test AUC: {best_fold[1]['test_auc']:.4f}")
    logger.info(f"Model ID: {results.get('best_model_id', 'N/A')} | Status: {results.get('status')}")
    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train NIFTY50 ML models on collected NIFTY data")
    parser.add_argument("--symbols", type=int, default=50, help="Number of NIFTY50 symbols")
    parser.add_argument("--train-years", type=int, default=3, help="Training window in years")
    parser.add_argument("--test-years", type=int, default=1, help="Test window in years")
    parser.add_argument("--model", type=str, default="xgboost", help="Model type: xgboost or lightgbm")
    parser.add_argument("--max-folds", type=int, default=5, help="Max walk-forward folds to train")
    parser.add_argument("--prob-long", type=float, default=0.52, help="Probability threshold to go long")
    parser.add_argument("--prob-short", type=float, default=0.48, help="Probability threshold to go short")
    parser.add_argument(
        "--pos-weight-mult",
        type=float,
        default=1.0,
        help="Multiplier applied to imbalance-based positive class weight",
    )
    parser.add_argument("--n-mc", type=int, default=500, help="Monte Carlo simulations for robustness")
    parser.add_argument(
        "--use-existing-data",
        action="store_true",
        default=False,
        help="Use existing data/nifty50_15y_ohlcv.csv instead of downloading",
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default="data/nifty50_15y_ohlcv.csv",
        help="Path to raw OHLCV CSV to use directly (skips collection if present)",
    )
    parser.add_argument(
        "--use-current-nifty50",
        action="store_true",
        help="Filter dataset to current NIFTY50 constituents",
    )

    args = parser.parse_args()

    results = run_complete_training_pipeline(
        num_symbols=args.symbols,
        train_years=args.train_years,
        test_years=args.test_years,
        model_type=args.model,
        max_folds=args.max_folds,
        prob_long=args.prob_long,
        prob_short=args.prob_short,
        pos_weight_mult=args.pos_weight_mult,
        n_monte_carlo=args.n_mc,
        use_existing_data=args.use_existing_data,
        raw_data_path=args.raw_data_path,
        use_current_nifty50=args.use_current_nifty50,
    )

    if results.get("status") == "completed":
        print("\n✓ Training pipeline completed successfully!")
        print(f"Best Model: {results['best_model_id']}")
        print(f"Best Test AUC: {results['stages']['model_training']['best_test_auc']:.4f}")
        if "stages" in results and "monte_carlo" in results["stages"]:
            mc = results["stages"]["monte_carlo"]
            if mc.get("status") == "success":
                print(
                    f"Monte Carlo Sharpe CI: {mc['sharpe_ci']} | Return CI: {mc['return_ci']}"
                )
    else:
        failed_stage = [k for k, v in results.get("stages", {}).items() if v.get("status") == "failed"]
        print(f"\n✗ Training pipeline failed at stage: {failed_stage[-1] if failed_stage else 'unknown'}")


if __name__ == "__main__":
    main()

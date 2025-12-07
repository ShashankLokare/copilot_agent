"""
Configuration management for the trading system.
Loads config from YAML files and provides typed access.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data sources."""
    csv_path: str = ""
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL"])
    timeframe: str = "1d"
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    enabled_indicators: List[str] = field(default_factory=lambda: [
        "SMA_20", "SMA_50", "RSI", "MACD", "ATR", "BOLLINGER"
    ])
    lookback_periods: Dict[str, int] = field(default_factory=lambda: {
        "price_lookback": 252,
        "vol_lookback": 20
    })


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    detector_type: str = "simple_rules"  # "simple_rules" or "ml"
    ml_model_path: Optional[str] = None
    trend_threshold: float = 0.5
    volatility_threshold: float = 1.0


@dataclass
class AlphaConfig:
    """Configuration for alpha models."""
    enabled_models: List[str] = field(default_factory=lambda: [
        "momentum", "mean_reversion", "breakout"
    ])
    model_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class SignalConfig:
    """Configuration for signal processing."""
    min_confidence: float = 0.5
    min_edge: float = 0.01  # 1% expected return
    merge_method: str = "weighted_average"


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    max_position_risk_pct: float = 1.0  # Max % equity at risk per trade
    max_daily_drawdown_pct: float = 5.0
    max_weekly_drawdown_pct: float = 10.0
    max_concurrent_positions: int = 10
    atr_stop_multiple: float = 2.0  # Stop loss at entry +/- N*ATR
    kill_switch_enabled: bool = True
    kill_switch_drawdown_pct: float = 20.0


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction."""
    method: str = "equal_weight"  # "equal_weight", "volatility_target", "risk_parity"
    diversification_method: str = "equal_weight"  # "equal_weight", "volatility_target"
    rebalance_frequency: str = "weekly"  # "daily", "weekly", "monthly"
    max_sector_exposure_pct: float = 30.0
    max_single_position_pct: float = 10.0
    max_position_size_pct: float = 10.0
    target_volatility: float = 0.15  # 15% annualized


@dataclass
class ExecutionConfig:
    """Configuration for execution."""
    execution_mode: str = "simulated"  # "simulated" or "live"
    slippage_bps: float = 2.0  # Basis points
    spread_bps: float = 1.0
    default_order_type: str = "MARKET"  # "MARKET" or "LIMIT"
    order_timeout_seconds: int = 300
    max_retries: int = 3
    retry_backoff_seconds: int = 5


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging."""
    log_level: str = "INFO"
    log_path: str = "logs/"
    metrics_frequency: str = "daily"  # "daily", "hourly", "minute"
    store_trades: bool = True
    trades_path: str = "data/trades.csv"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    operation_mode: str = "PAPER"  # "LIVE", "PAPER", "BACKTEST"
    run_frequency: str = "daily"  # "minute", "hourly", "daily"
    scheduled_time: str = "16:00"  # HH:MM in UTC
    enabled_markets: List[str] = field(default_factory=lambda: ["stocks"])


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    
    @classmethod
    def load_from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        
        # Instantiate nested dataclasses properly
        kwargs = {}
        if "data" in config_dict:
            kwargs["data"] = DataConfig(**config_dict["data"])
        if "features" in config_dict:
            kwargs["features"] = FeatureConfig(**config_dict["features"])
        if "regime" in config_dict:
            kwargs["regime"] = RegimeConfig(**config_dict["regime"])
        if "alpha" in config_dict:
            kwargs["alpha"] = AlphaConfig(**config_dict["alpha"])
        if "signals" in config_dict:
            kwargs["signals"] = SignalConfig(**config_dict["signals"])
        if "risk" in config_dict:
            kwargs["risk"] = RiskConfig(**config_dict["risk"])
        if "portfolio" in config_dict:
            kwargs["portfolio"] = PortfolioConfig(**config_dict["portfolio"])
        if "execution" in config_dict:
            kwargs["execution"] = ExecutionConfig(**config_dict["execution"])
        if "monitoring" in config_dict:
            kwargs["monitoring"] = MonitoringConfig(**config_dict["monitoring"])
        if "orchestrator" in config_dict:
            kwargs["orchestrator"] = OrchestratorConfig(**config_dict["orchestrator"])
        
        return cls(**kwargs)
    
    @classmethod
    def load_from_file(cls, config_path: str) -> "Config":
        """Load configuration from file (auto-detect format)."""
        path = Path(config_path)
        if path.suffix in [".yaml", ".yml"]:
            return cls.load_from_yaml(config_path)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "data": self.data.__dict__,
            "features": self.features.__dict__,
            "regime": self.regime.__dict__,
            "alpha": self.alpha.__dict__,
            "signals": self.signals.__dict__,
            "risk": self.risk.__dict__,
            "portfolio": self.portfolio.__dict__,
            "execution": self.execution.__dict__,
            "monitoring": self.monitoring.__dict__,
            "orchestrator": self.orchestrator.__dict__,
        }

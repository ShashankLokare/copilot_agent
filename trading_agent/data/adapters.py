"""
Data source adapters for market data.
Provides a common interface for different data sources.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
from pathlib import Path
import pytz

from utils.types import Bar, MarketState


class DataAdapter(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> List[Bar]:
        """Fetch historical OHLCV bars for a symbol."""
        pass
    
    @abstractmethod
    def fetch_latest_price(self, symbol: str) -> float:
        """Fetch the latest price for a symbol."""
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass


class CSVAdapter(DataAdapter):
    """Adapter for CSV data files."""
    
    def __init__(self, csv_path: str):
        """
        Initialize CSV adapter.
        
        Args:
            csv_path: Path to CSV file. Expected columns: symbol, timestamp, open, high, low, close, volume
        """
        self.csv_path = Path(csv_path)
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load and prepare CSV data."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Ensure timestamp is UTC
        if self.df['timestamp'].dt.tz is None:
            self.df['timestamp'] = self.df['timestamp'].dt.tz_localize(pytz.UTC)
        else:
            self.df['timestamp'] = self.df['timestamp'].dt.tz_convert(pytz.UTC)
        
        self.df = self.df.sort_values('timestamp')
    
    def fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> List[Bar]:
        """Fetch historical bars for a symbol."""
        if self.df is None:
            raise RuntimeError("Data not loaded")
        
        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = pytz.UTC.localize(start_date)
        if end_date.tzinfo is None:
            end_date = pytz.UTC.localize(end_date)
        
        # Filter by symbol and date range
        mask = (
            (self.df['symbol'] == symbol) &
            (self.df['timestamp'] >= start_date) &
            (self.df['timestamp'] <= end_date)
        )
        filtered = self.df[mask]
        
        if filtered.empty:
            return []
        
        bars = []
        for _, row in filtered.iterrows():
            bar = Bar(
                symbol=row['symbol'],
                timestamp=row['timestamp'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            bars.append(bar)
        
        return bars
    
    def fetch_latest_price(self, symbol: str) -> float:
        """Fetch latest price for a symbol."""
        if self.df is None:
            raise RuntimeError("Data not loaded")
        
        mask = self.df['symbol'] == symbol
        filtered = self.df[mask].sort_values('timestamp')
        
        if filtered.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        return float(filtered.iloc[-1]['close'])
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        if self.df is None:
            raise RuntimeError("Data not loaded")
        
        return self.df['symbol'].unique().tolist()


class RESTAPIAdapter(DataAdapter):
    """Adapter for REST API data sources (placeholder)."""
    
    def __init__(self, api_endpoint: str, api_key: Optional[str] = None):
        """
        Initialize REST API adapter.
        
        Args:
            api_endpoint: Base URL for the API
            api_key: Optional API key for authentication
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.symbols: List[str] = []
    
    def fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> List[Bar]:
        """
        Fetch historical bars from REST API.
        
        This is a placeholder implementation.
        Subclass and implement actual API calls.
        """
        raise NotImplementedError("Subclass and implement fetch_bars()")
    
    def fetch_latest_price(self, symbol: str) -> float:
        """
        Fetch latest price from REST API.
        
        This is a placeholder implementation.
        """
        raise NotImplementedError("Subclass and implement fetch_latest_price()")
    
    def get_symbols(self) -> List[str]:
        """Get available symbols."""
        return self.symbols


class MarketDataProvider:
    """
    Main interface for market data.
    Wraps a DataAdapter and provides convenient methods.
    """
    
    def __init__(self, adapter: DataAdapter):
        """
        Initialize with a data adapter.
        
        Args:
            adapter: DataAdapter implementation (e.g., CSVAdapter)
        """
        self.adapter = adapter
        self._bar_cache: Dict[str, List[Bar]] = {}
    
    def get_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
        use_cache: bool = True
    ) -> List[Bar]:
        """Get bars for a symbol."""
        cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        
        if use_cache and cache_key in self._bar_cache:
            return self._bar_cache[cache_key]
        
        bars = self.adapter.fetch_bars(symbol, start_date, end_date, timeframe)
        
        if use_cache:
            self._bar_cache[cache_key] = bars
        
        return bars
    
    def get_market_state(
        self,
        symbol: str,
        timestamp: datetime,
        lookback_bars: int = 50
    ) -> MarketState:
        """Get current market state for a symbol."""
        price = self.adapter.fetch_latest_price(symbol)
        
        # Fetch recent bars for volatility calculation
        # In production, these would come from live data
        bars = self.adapter.fetch_bars(
            symbol,
            timestamp.replace(day=1),  # Start of month (simple approach)
            timestamp
        )
        
        if bars:
            recent_bars = bars[-lookback_bars:]
            volatility = self._calculate_volatility(recent_bars)
        else:
            volatility = 0.0
            recent_bars = []
        
        return MarketState(
            symbol=symbol,
            timestamp=timestamp,
            current_price=price,
            bid=price * 0.9999,  # Placeholder bid/ask
            ask=price * 1.0001,
            volume=1000.0,  # Placeholder
            volatility=volatility,
            bars=recent_bars
        )
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return self.adapter.get_symbols()
    
    @staticmethod
    def _calculate_volatility(bars: List[Bar]) -> float:
        """Calculate annualized volatility from bars."""
        if len(bars) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(bars)):
            ret = (bars[i].close - bars[i-1].close) / bars[i-1].close
            returns.append(ret)
        
        if not returns:
            return 0.0
        
        import numpy as np
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)  # Annualize
        
        return float(annual_vol)

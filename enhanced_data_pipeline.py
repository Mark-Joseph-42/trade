"""Enhanced data pipeline for financial time series data."""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import VolumeWeightedAveragePrice


@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    symbols: List[str]
    timeframes: List[str]
    start_date: str
    end_date: str
    data_dir: str = "data"
    max_retries: int = 3
    cache_dir: str = "cache"
    
    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)


class FeatureEngineer:
    """Feature engineering for financial time series data."""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
            
        # Add all standard technical analysis features
        df = add_all_ta_features(
            df, 
            open="open", 
            high="high", 
            low="low", 
            close="close", 
            volume="volume",
            fillna=True
        )
        
        # Add custom indicators
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # RSI
        rsi = RSIIndicator(close=df['close'])
        df['rsi'] = rsi.rsi()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        
        # ATR
        atr = AverageTrueRange(
            high=df['high'], 
            low=df['low'], 
            close=df['close']
        )
        df['atr'] = atr.average_true_range()
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )
        df['vwap'] = vwap.volume_weighted_average_price()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX
        adx = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
        df['adx'] = adx.adx()
        
        # Add price returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Add volatility
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Clean up any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataframe."""
        df = df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
            
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Market session (simplified)
        df['market_session'] = pd.cut(
            df['hour'],
            bins=[-1, 4, 8, 12, 16, 20, 24],
            labels=['overnight', 'asian', 'european', 'london', 'new_york', 'evening']
        )
        
        # One-hot encode the market session
        session_dummies = pd.get_dummies(df['market_session'], prefix='session')
        df = pd.concat([df, session_dummies], axis=1)
        
        return df


class DataPipeline:
    """Enhanced data pipeline for financial time series data."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer()
    
    def fetch_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance with retry logic."""
        cache_file = os.path.join(
            self.config.cache_dir, 
            f"{symbol.replace('/', '_')}_{timeframe}_{self.config.start_date}_{self.config.end_date}.parquet"
        )
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                print(f"Error loading cached data: {e}")
        
        # Download data from yfinance
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval=timeframe,
                auto_adjust=True
            )
            if df.empty:
                # Try alternative method if the first one fails
                df = yf.download(
                    symbol,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval=timeframe,
                    progress=False
                )
            
            if df.empty:
                raise ValueError(f"No data found for {symbol} {timeframe}")
                
        except Exception as e:
            print(f"Error downloading data for {symbol} {timeframe}: {e}")
            raise
        
        # Standardize column names
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            'Dividends': 'dividends',
            'Stock Splits': 'splits'
        })
        
        # Save to cache
        if not df.empty:
            df.to_parquet(cache_file)
        else:
            raise ValueError(f"Empty dataframe for {symbol} {timeframe}")
        
        return df
    
    def process_symbol(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Process data for a single symbol across all timeframes."""
        result = {}
        
        for tf in self.config.timeframes:
            try:
                print(f"Processing {symbol} {tf}...")
                
                # Fetch data
                df = self.fetch_data(symbol, tf)
                
                # Add technical indicators
                df = self.feature_engineer.add_technical_indicators(df)
                
                # Add time features
                df = self.feature_engineer.add_time_features(df)
                
                # Store in results
                result[tf] = df
                
            except Exception as e:
                print(f"Error processing {symbol} {tf}: {e}")
                continue
                
        return result
    
    def run(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Run the data pipeline for all symbols and timeframes."""
        results = {}
        
        for symbol in self.config.symbols:
            try:
                print(f"\nProcessing {symbol}...")
                results[symbol] = self.process_symbol(symbol)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
                
        return results


def main():
    # Example usage
    config = DataConfig(
        symbols=["AAPL", "MSFT", "GOOGL"],
        timeframes=["1d", "1h"],
        start_date="2020-01-01",
        end_date="2023-01-01"
    )
    
    pipeline = DataPipeline(config)
    data = pipeline.run()
    
    # Example: Access 1d data for AAPL
    aapl_daily = data["AAPL"]["1d"]
    print("\nSample data for AAPL (Daily):")
    print(aapl_daily[["open", "high", "low", "close", "volume", "rsi", "macd"]].tail())


if __name__ == "__main__":
    main()

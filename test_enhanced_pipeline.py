"""Test script for the enhanced data pipeline."""
import unittest
import pandas as pd
import numpy as np
from enhanced_data_pipeline import DataConfig, DataPipeline, FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test the FeatureEngineer class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 0,
            'low': 0,
            'close': 0,
            'volume': 1000000 + np.random.randint(-100000, 100000, 100)
        })
        
        # Set high/low/close based on open with some randomness
        self.df['high'] = self.df['open'] + np.abs(np.random.randn(100)) * 2
        self.df['low'] = self.df['open'] - np.abs(np.random.randn(100)) * 2
        self.df['close'] = (self.df['open'] + self.df['high'] + self.df['low']) / 3
        self.df.index = dates
        
        self.engineer = FeatureEngineer()
    
    def test_add_technical_indicators(self):
        """Test adding technical indicators."""
        df = self.engineer.add_technical_indicators(self.df)
        
        # Check that required columns were added
        expected_columns = ['rsi', 'macd', 'macd_signal', 'atr', 'vwap', 'returns']
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Check for NaN values
        self.assertFalse(df[expected_columns].isna().any().any())
    
    def test_add_time_features(self):
        """Test adding time-based features."""
        df = self.engineer.add_time_features(self.df)
        
        # Check that time features were added
        expected_columns = ['hour', 'day_of_week', 'day_of_month', 'month']
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Check market session one-hot encoding
        self.assertTrue(any(col.startswith('session_') for col in df.columns))


class TestDataPipeline(unittest.TestCase):
    """Test the DataPipeline class."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = DataConfig(
            symbols=["AAPL"],  # Single symbol for testing
            timeframes=["1d"],  # Single timeframe for testing
            start_date="2023-01-01",
            end_date="2023-02-01",
            cache_dir="test_cache"
        )
        
        # Create test cache directory
        import os
        os.makedirs("test_cache", exist_ok=True)
        
        self.pipeline = DataPipeline(self.config)
    
    def test_fetch_data(self):
        """Test data fetching functionality."""
        df = self.pipeline.fetch_data("AAPL", "1d")
        
        # Check basic structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, df.columns)
    
    def test_process_symbol(self):
        """Test processing a single symbol."""
        result = self.pipeline.process_symbol("AAPL")
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("1d", result)
        
        # Check that technical indicators were added
        df = result["1d"]
        self.assertIn("rsi", df.columns)
        self.assertIn("macd", df.columns)
        
        # Check that time features were added
        self.assertIn("hour", df.columns)
        self.assertIn("day_of_week", df.columns)
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        try:
            shutil.rmtree("test_cache")
        except Exception as e:
            print(f"Error cleaning up test cache: {e}")


if __name__ == "__main__":
    unittest.main()

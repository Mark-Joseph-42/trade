"""Test script for the enhanced trading environment."""
import unittest
import numpy as np
import pandas as pd
from enhanced_rl_env import EnhancedTradingEnv


class TestEnhancedTradingEnv(unittest.TestCase):
    """Test the EnhancedTradingEnv class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create sample OHLCV data with technical indicators
        np.random.seed(42)
        n = 1000
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
        
        # Generate random walk for prices
        returns = np.random.randn(n) * 0.01
        prices = 100 * np.cumprod(1 + returns)
        
        # Create OHLCV data
        cls.df = pd.DataFrame({
            'open': prices * (1 + (np.random.rand(n) - 0.5) * 0.01),
            'high': prices * (1 + np.random.rand(n) * 0.02),
            'low': prices * (1 - np.random.rand(n) * 0.02),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n)
        }, index=dates)
        
        # Add some technical indicators
        cls.df['rsi'] = 50 + 30 * np.sin(np.linspace(0, 10, n))  # Oscillating RSI
        cls.df['macd'] = np.cumsum(np.random.randn(n)) * 0.1
        cls.df['atr'] = np.ones(n) * 2.5
        cls.df['returns'] = np.diff(cls.df['close'], prepend=cls.df['close'].iloc[0]) / cls.df['close'].iloc[0]
    
    def setUp(self):
        """Set up the test environment."""
        self.env = EnhancedTradingEnv(
            price_df=self.df,
            window_size=64,
            initial_balance=10000.0,
            fee_rate=0.001,
            mdd_cap=0.10,
            use_gaf=True
        )
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.initial_balance, 10000.0)
        self.assertEqual(self.env.window_size, 64)
        self.assertEqual(self.env.action_space_n, 3)
        
        # Check observation shape
        obs = self.env.reset()
        self.assertEqual(obs.shape, (len(self.env.feature_columns), 32, 32))
    
    def test_reset(self):
        """Test environment reset."""
        obs = self.env.reset()
        self.assertEqual(self.env.step_index, self.env.window_size)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.balance, self.env.initial_balance)
        self.assertEqual(self.env.equity, self.env.initial_balance)
        self.assertEqual(len(self.env.equity_history), 1)
        self.assertEqual(len(self.env.returns_history), 0)
        self.assertEqual(obs.shape, (len(self.env.feature_columns), 32, 32))
    
    def test_buy_sell_cycle(self):
        """Test a complete buy-sell cycle."""
        # Reset environment
        obs = self.env.reset()
        
        # Take BUY action
        obs, reward, done, info = self.env.step(1)  # BUY
        self.assertEqual(self.env.position, 1)
        self.assertGreater(self.env.position_size, 0)
        self.assertGreater(self.env.entry_price, 0)
        self.assertLess(self.env.balance, self.env.initial_balance)
        
        # Take SELL action
        obs, reward, done, info = self.env.step(2)  # SELL
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.position_size, 0.0)
        self.assertEqual(self.env.entry_price, 0.0)
    
    def test_hold_action(self):
        """Test HOLD action."""
        obs = self.env.reset()
        
        # Take BUY action first
        obs, _, _, _ = self.env.step(1)  # BUY
        position_size = self.env.position_size
        
        # Take HOLD action
        obs, reward, done, info = self.env.step(0)  # HOLD
        self.assertEqual(self.env.position, 1)
        self.assertEqual(self.env.position_size, position_size)
    
    def test_mdd_violation(self):
        """Test maximum drawdown violation."""
        # Create a new environment with a very low MDD cap
        env = EnhancedTradingEnv(
            price_df=self.df,
            window_size=64,
            initial_balance=10000.0,
            fee_rate=0.001,
            mdd_cap=0.01,  # Very low MDD cap for testing
            use_gaf=True
        )
        
        # Reset environment
        obs = env.reset()
        
        # Take BUY action
        obs, reward, done, info = env.step(1)  # BUY
        
        # Force a large drawdown
        env.equity = env.initial_balance * 0.95  # 5% drawdown
        env.balance = env.initial_balance * 0.95
        
        # Take another step (should trigger MDD violation)
        obs, reward, done, info = env.step(0)  # HOLD
        
        # Check if episode was terminated due to MDD violation
        self.assertTrue(done)
        self.assertEqual(info["status"], "mdd_violation")
    
    def test_trade_history(self):
        """Test trade history recording."""
        # Reset environment
        obs = self.env.reset()
        
        # Take some actions
        self.env.step(1)  # BUY
        self.env.step(0)  # HOLD
        self.env.step(2)  # SELL
        
        # Get trade history
        trade_history = self.env.get_trade_history()
        
        # Check trade history
        self.assertEqual(len(trade_history), 2)  # One BUY, one SELL
        self.assertEqual(trade_history.iloc[0]["action"], "LONG")
        self.assertEqual(trade_history.iloc[1]["action"], "CLOSE")


if __name__ == "__main__":
    unittest.main()

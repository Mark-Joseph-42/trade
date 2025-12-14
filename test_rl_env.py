"""Test script for the RL trading environment."""
import os
import sys
import numpy as np
import pandas as pd
import unittest
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_env import TradingEnv

def create_test_data(days=100):
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, days)
    prices = base_price * (1 + np.cumsum(returns)).cumprod()
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000, 10000, days)
    }, index=dates)
    
    return df

class TestTradingEnv(unittest.TestCase):
    """Test cases for TradingEnv class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data = create_test_data()
        self.env = TradingEnv(
            price_df=self.test_data,
            window_size=10,
            image_size=32,
            initial_balance=10000.0,
            fee_rate=0.001,
            mdd_cap=0.10
        )
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.initial_balance, 10000.0)
        self.assertEqual(self.env.balance, 10000.0)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.step_index, 0)
        self.assertEqual(self.env.action_space_n, 3)
    
    def test_reset(self):
        """Test environment reset."""
        # Take a step to change the state
        state = self.env.reset()
        
        # Check if state is a numpy array with correct shape
        self.assertIsInstance(state, np.ndarray)
        # Expected shape is (2, image_size, image_size) for GAF encoding
        self.assertEqual(state.shape, (2, 32, 32))  # (channels, height, width)
        
        # Check if environment is reset to initial state
        self.assertEqual(self.env.balance, 10000.0)
        self.assertEqual(self.env.position, 0)
        # After reset, step_index should be at window_size to have enough data
        self.assertEqual(self.env.step_index, self.env.window_size)
    
    def test_step(self):
        """Test taking a step in the environment."""
        self.env.reset()
        
        # Take a step with action 1 (buy)
        state, reward, done, info = self.env.step(1)
        
        # Check return types
        self.assertIsInstance(state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
        # Check if position was opened
        self.assertNotEqual(self.env.position, 0)
        
        # Take another step to close position
        state, reward, done, info = self.env.step(0)  # Close position
        self.assertEqual(self.env.position, 0)
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        self.env.reset()
        
        # Take a buy action
        _, reward1, _, _ = self.env.step(1)  # Buy
        
        # Take a sell action
        _, reward2, _, _ = self.env.step(0)  # Close position
        
        # Check if rewards are calculated
        self.assertIsInstance(reward1, float)
        self.assertIsInstance(reward2, float)
        
        # Check if rewards are different (should be based on PnL)
        self.assertNotEqual(reward1, reward2)
    
    def test_termination_conditions(self):
        """Test episode termination conditions."""
        self.env.reset()
        
        # Run through all steps
        done = False
        steps = 0
        max_steps = len(self.test_data) - self.env.window_size - 1
        
        while not done and steps < max_steps:
            _, _, done, _ = self.env.step(np.random.randint(0, 3))  # Random action
            steps += 1
        
        # Should terminate when we reach the end of the data
        self.assertTrue(done)
        self.assertEqual(steps, max_steps)

if __name__ == "__main__":
    unittest.main()

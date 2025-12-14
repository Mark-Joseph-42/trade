import pandas as pd
import numpy as np
from trading_env import GAFTradingEnv

def create_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    return pd.DataFrame({
        'date': dates,
        'open': close_prices - np.random.uniform(0.1, 1, 1000),
        'high': close_prices + np.random.uniform(0.1, 1, 1000),
        'low': close_prices - np.random.uniform(0.1, 1, 1000),
        'close': close_prices,
        'volume': np.random.randint(100, 10000, 1000)
    }).set_index('date')

def test_environment():
    print("Testing GAFTradingEnv...")
    df = create_sample_data()
    env = GAFTradingEnv(df, window_size=50)
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Done={done}")
        print(f"  Equity: {info['equity']:.2f}, Position: {info['position']}, Drawdown: {info['drawdown']*100:.2f}%")

if __name__ == "__main__":
    test_environment()

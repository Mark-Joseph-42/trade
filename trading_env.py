import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Tuple, Dict, Any
from gaf_encoder import gaf_from_series

class GAFTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df: pd.DataFrame, window_size: int = 50):
        super(GAFTradingEnv, self).__init__()
        self.df = df
        self.window_size = window_size
        self.current_step = window_size
        self.max_steps = len(df) - 1
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: GAF image
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(1, window_size, window_size), 
            dtype=np.float32
        )
        
        # Trading state
        self.position = 0  # 0=no position, 1=long
        self.entry_price = 0
        self.equity = 10000  # Starting capital
        self.max_equity = self.equity
        self.drawdown = 0
        
    def _get_observation(self) -> np.ndarray:
        # Get window of close prices
        window = self.df['close'].iloc[
            self.current_step - self.window_size : self.current_step
        ].values
        
        # Convert to GAF image
        gaf_image = gaf_from_series(window)
        
        # Ensure the GAF image has the correct dimensions
        if gaf_image.ndim == 2:
            # If 2D, add channel dimension
            gaf_image = gaf_image[np.newaxis, ...]
        
        # Ensure the image has the correct shape (1, window_size, window_size)
        if gaf_image.shape[0] != 1:
            gaf_image = gaf_image.reshape(1, self.window_size, self.window_size)
            
        return gaf_image.astype(np.float32)
    
    def _calculate_reward(self, current_price: float) -> float:
        # Calculate raw PnL
        if self.position == 1:  # Long position
            raw_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            raw_pnl = 0  # No position, no PnL
            
        # Calculate drawdown
        self.equity = self.equity * (1 + raw_pnl)
        self.max_equity = max(self.max_equity, self.equity)
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        self.drawdown = max(self.drawdown, current_drawdown)
        
        # Penalize drawdown
        drawdown_penalty = 0.5 * self.drawdown
        reward = raw_pnl - drawdown_penalty
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        current_price = self.df['close'].iloc[self.current_step]
        done = False
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
        
        # Calculate reward
        reward = self._calculate_reward(current_price)
        
        # Check for termination
        self.current_step += 1
        if self.current_step >= self.max_steps or self.drawdown > 0.10:
            done = True
            
        # Get next observation
        obs = self._get_observation()
        
        info = {
            'equity': self.equity,
            'position': self.position,
            'drawdown': self.drawdown
        }
        
        return obs, reward, done, info
    
    def reset(self) -> np.ndarray:
        self.current_step = self.window_size
        self.position = 0
        self.equity = 10000
        self.max_equity = self.equity
        self.drawdown = 0
        return self._get_observation()
    
    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Equity: {self.equity:.2f}, '
              f'Position: {self.position}, Drawdown: {self.drawdown*100:.2f}%')

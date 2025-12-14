"""Enhanced RL Environment for Trading with Support for Advanced Features."""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List

from gaf_encoder import gaf_from_multichannel
from risk_module import (
    compute_max_drawdown,
    dynamic_position_size,
    kill_switch_triggered,
)


class EnhancedTradingEnv:
    """Enhanced trading environment with support for advanced features."""
    
    def __init__(
        self,
        price_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        window_size: int = 64,
        image_size: int = 32,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
        mdd_cap: float = 0.10,
        volatility_lookback: int = 20,
        reward_mdd_penalty: float = 1.0,
        use_gaf: bool = True,
        normalize: bool = True,
    ):
        """Initialize the enhanced trading environment.
        
        Args:
            price_df: DataFrame containing OHLCV and technical indicators
            feature_columns: List of column names to use as features
            window_size: Number of time steps in the observation window
            image_size: Size of the GAF image (if use_gaf is True)
            initial_balance: Initial account balance
            fee_rate: Trading fee rate
            mdd_cap: Maximum allowed drawdown before episode termination
            volatility_lookback: Lookback period for volatility calculation
            reward_mdd_penalty: Penalty factor for drawdown in reward
            use_gaf: Whether to use GAF for state representation
            normalize: Whether to normalize the observation space
        """
        # Validate input data
        required_columns = ["open", "high", "low", "close", "volume"]
        if not all(col in price_df.columns for col in required_columns):
            raise ValueError("price_df must contain columns: open, high, low, close, volume")
            
        self.df = price_df.copy()
        self.window_size = int(window_size)
        self.image_size = int(image_size)
        self.initial_balance = float(initial_balance)
        self.fee_rate = float(fee_rate)
        self.mdd_cap = float(mdd_cap)
        self.volatility_lookback = int(volatility_lookback)
        self.reward_mdd_penalty = float(reward_mdd_penalty)
        self.use_gaf = bool(use_gaf)
        self.normalize = bool(normalize)
        
        # Set feature columns
        if feature_columns is None:
            # Default to OHLCV + common technical indicators
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'macd_signal', 'atr', 'vwap',
                'stoch_k', 'stoch_d', 'adx', 'returns'
            ]
            # Only keep columns that exist in the dataframe
            self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
        else:
            self.feature_columns = list(feature_columns)
        
        # Initialize state
        self.reset()
    
    @property
    def action_space_n(self) -> int:
        """Number of possible actions."""
        return 3  # HOLD, BUY, SELL
    
    @property
    def observation_shape(self) -> tuple:
        """Shape of the observation space."""
        if self.use_gaf:
            # GAF returns (channels, height, width)
            return (len(self.feature_columns), self.image_size, self.image_size)
        else:
            # Raw features: (window_size, num_features)
            return (self.window_size, len(self.feature_columns))
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.step_index = self.window_size
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.equity_history = [self.initial_balance]
        self.returns_history = []
        self.trade_history = []
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        start = self.step_index - self.window_size
        end = self.step_index
        window = self.df.iloc[start:end][self.feature_columns]
        
        # Convert to numpy array
        obs = window.to_numpy(dtype=np.float32)
        
        # Normalize if requested
        if self.normalize:
            obs = self._normalize_observation(obs)
        
        # Convert to GAF if requested
        if self.use_gaf:
            # Transpose to (features, window_size) for GAF
            obs = obs.T
            obs = gaf_from_multichannel(obs, image_size=self.image_size)
        
        return obs
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize the observation using running statistics."""
        # Simple min-max normalization per feature
        min_vals = np.min(obs, axis=0, keepdims=True)
        max_vals = np.max(obs, axis=0, keepdims=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        return (obs - min_vals) / range_vals
    
    def _update_equity(self, price: float) -> None:
        """Update the equity value based on current position."""
        position_value = self.position_size * price
        self.equity = self.balance + position_value
        self.equity_history.append(self.equity)
    
    def _calculate_reward(self, prev_equity: float, current_price: float) -> float:
        """Calculate the reward for the current step."""
        # Simple PnL based reward
        current_equity = self.balance + (self.position_size * current_price)
        reward = (current_equity - prev_equity) / prev_equity  # Return as percentage
        
        # Add penalty for drawdown
        if len(self.equity_history) > 1:
            max_equity = max(self.equity_history)
            drawdown = (max_equity - current_equity) / max_equity
            reward -= self.reward_mdd_penalty * drawdown
        
        return float(reward)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        if action < 0 or action >= self.action_space_n:
            raise ValueError(f"Invalid action {action}; must be in [0, {self.action_space_n - 1}]")
        
        # Check if episode is done
        if self.step_index >= len(self.df) - 1:
            obs = self._get_observation()
            return obs, 0.0, True, {"status": "episode_end", "equity": self.equity}
        
        # Get current price and previous equity
        row = self.df.iloc[self.step_index]
        price = float(row["close"])
        prev_equity = self.equity
        
        # Execute trade
        if action == 1:  # BUY
            if self.position <= 0:  # Only buy if not already long
                # Close short position if any
                if self.position < 0:
                    self._close_position(price)
                # Open long position
                self._open_position(1, price)
        elif action == 2:  # SELL
            if self.position >= 0:  # Only sell if not already short
                # Close long position if any
                if self.position > 0:
                    self._close_position(price)
                # Open short position
                self._open_position(-1, price)
        # HOLD: Do nothing
        
        # Update equity and calculate reward
        self._update_equity(price)
        reward = self._calculate_reward(prev_equity, price)
        
        # Update step index
        self.step_index += 1
        
        # Get next observation
        obs = self._get_observation()
        
        # Check for termination conditions
        done = False
        info = {
            "equity": self.equity,
            "position": self.position,
            "price": price,
            "step": self.step_index,
            "status": "running"
        }
        
        # Check for max drawdown violation
        if self._check_mdd_violation():
            done = True
            info["status"] = "mdd_violation"
            reward -= 1.0  # Additional penalty for MDD violation
        
        return obs, reward, done, info
    
    def _open_position(self, direction: int, price: float) -> None:
        """Open a new position."""
        assert direction in (-1, 1), "Direction must be -1 (short) or 1 (long)"
        
        # Calculate position size based on volatility
        if len(self.returns_history) >= self.volatility_lookback:
            recent = np.array(self.returns_history[-self.volatility_lookback:], dtype=np.float32)
            volatility = float(np.std(recent))
        else:
            volatility = 0.1  # Default value
        
        # Calculate position size
        max_risk = 0.02  # Max 2% risk per trade
        position_value = self.equity * max_risk / (volatility + 1e-8)
        position_value = min(position_value, self.equity * 0.99)  # Leave some margin for fees
        
        # Update position
        self.position = direction
        self.position_size = position_value / price
        self.entry_price = price
        self.balance -= position_value  # Reserve the funds
        
        # Record trade
        self.trade_history.append({
            "step": self.step_index,
            "action": "LONG" if direction > 0 else "SHORT",
            "price": price,
            "size": self.position_size,
            "value": position_value,
            "balance": self.balance,
            "equity": self.equity
        })
    
    def _close_position(self, price: float) -> None:
        """Close the current position."""
        if self.position == 0:
            return
        
        # Calculate PnL
        pnl = self.position * self.position_size * (price - self.entry_price)
        pnl -= abs(pnl) * self.fee_rate  # Apply fees
        
        # Update balance and reset position
        self.balance += self.position_size * self.entry_price + pnl
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        
        # Record trade
        self.trade_history.append({
            "step": self.step_index,
            "action": "CLOSE",
            "price": price,
            "pnl": pnl,
            "balance": self.balance,
            "equity": self.equity
        })
    
    def _check_mdd_violation(self) -> bool:
        """Check if maximum drawdown has been violated."""
        if len(self.equity_history) < 2:
            return False
        
        max_equity = max(self.equity_history)
        current_equity = self.equity_history[-1]
        drawdown = (max_equity - current_equity) / max_equity
        
        return drawdown >= self.mdd_cap
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get the trade history as a DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history)
    
    def get_equity_history(self) -> pd.Series:
        """Get the equity history as a Series."""
        return pd.Series(self.equity_history, name="equity")

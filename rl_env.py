import numpy as np
import pandas as pd

from typing import Tuple

from gaf_encoder import gaf_from_multichannel
from risk_module import (
    compute_max_drawdown,
    dynamic_position_size,
    kill_switch_triggered,
)


class TradingEnv:
    def __init__(
        self,
        price_df: pd.DataFrame,
        window_size: int = 64,
        image_size: int = 32,
        initial_balance: float = 1000.0,
        fee_rate: float = 0.001,
        mdd_cap: float = 0.10,
        volatility_lookback: int = 20,
        reward_mdd_penalty: float = 1.0,
    ):
        if not {"open", "high", "low", "close", "volume"}.issubset(price_df.columns):
            raise ValueError("price_df must contain columns: open, high, low, close, volume")
        self.df = price_df.reset_index(drop=True)
        self.window_size = int(window_size)
        self.image_size = int(image_size)
        self.initial_balance = float(initial_balance)
        self.fee_rate = float(fee_rate)
        self.mdd_cap = float(mdd_cap)
        self.volatility_lookback = int(volatility_lookback)
        self.reward_mdd_penalty = float(reward_mdd_penalty)

        self.step_index = 0
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.equity_history = []
        self.returns_history = []

    @property
    def action_space_n(self) -> int:
        return 3

    def reset(self) -> np.ndarray:
        self.step_index = self.window_size
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.equity_history = [self.initial_balance]
        self.returns_history = []
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        start = self.step_index - self.window_size
        end = self.step_index
        window = self.df.iloc[start:end]
        close = window["close"].to_numpy(dtype=np.float32)
        volume = window["volume"].to_numpy(dtype=np.float32)
        gaf = gaf_from_multichannel([close, volume], image_size=self.image_size)
        return gaf

    def _update_equity(self, price: float) -> None:
        position_value = self.position_size * price
        self.equity = self.balance + position_value
        self.equity_history.append(self.equity)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if action < 0 or action >= self.action_space_n:
            raise ValueError(f"Invalid action {action}; must be in [0, {self.action_space_n - 1}]")
        if self.step_index >= len(self.df) - 1:
            return self._get_observation(), 0.0, True, {}

        row = self.df.iloc[self.step_index]
        price = float(row["close"])

        if action == 0:
            target_position = 0
        elif action == 1:
            target_position = 1
        else:
            target_position = -1

        prev_equity = self.equity
        if len(self.returns_history) >= self.volatility_lookback:
            recent = np.array(self.returns_history[-self.volatility_lookback :], dtype=np.float32)
            volatility = float(np.std(recent))
        else:
            volatility = 0.0

        target_fraction = dynamic_position_size(
            equity=self.equity,
            volatility=volatility,
        )

        target_position_value = target_position * target_fraction * self.equity
        target_size = target_position_value / price if price > 0 else 0.0
        trade_size_change = target_size - self.position_size
        trade_notional = abs(trade_size_change) * price
        fee = trade_notional * self.fee_rate

        self.balance -= trade_size_change * price
        self.balance -= fee
        self.position = target_position
        self.position_size = target_size

        self.step_index += 1
        next_price = float(self.df.iloc[self.step_index]["close"])
        self._update_equity(next_price)

        step_return = (self.equity - prev_equity) / max(prev_equity, 1e-8)
        self.returns_history.append(step_return)

        mdd = compute_max_drawdown(np.array(self.equity_history, dtype=np.float32))
        reward = float(step_return - self.reward_mdd_penalty * max(0.0, mdd))

        done_kill = kill_switch_triggered(
            np.array(self.equity_history, dtype=np.float32),
            mdd_cap=self.mdd_cap,
        )
        done_end = self.step_index >= len(self.df) - 1
        done = bool(done_kill or done_end)

        info = {
            "equity": self.equity,
            "balance": self.balance,
            "position": self.position,
            "position_size": self.position_size,
            "price": next_price,
            "step_return": step_return,
            "mdd": mdd,
            "done_kill": done_kill,
        }

        obs = self._get_observation()
        return obs, reward, done, info

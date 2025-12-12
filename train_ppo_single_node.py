import os
from typing import Tuple

import numpy as np

from config import EXCHANGE_ID, FEE_RATE, KLINE_INTERVAL, SIM_STARTING_BALANCE
from data_pipeline import get_or_download_ohlcv, split_train_test
from ppo_cnn_agent import PPOAgent, PPOConfig
from rl_env import TradingEnv
from risk_module import compute_max_drawdown


def steps_per_year_from_timeframe(timeframe: str) -> int:
    if timeframe.endswith("m"):
        m = int(timeframe[:-1])
        return int((60 / m) * 24 * 365)
    if timeframe.endswith("h"):
        h = int(timeframe[:-1])
        return int((24 / h) * 365)
    if timeframe.endswith("d"):
        d = int(timeframe[:-1])
        return int(365 / d)
    return 365


def annualized_sharpe(returns: np.ndarray, steps_per_year: int) -> float:
    r = np.asarray(returns, dtype=np.float32).reshape(-1)
    if r.size < 2:
        return 0.0
    mean = float(r.mean())
    std = float(r.std())
    if std == 0.0:
        return 0.0
    return float((mean / std) * np.sqrt(steps_per_year))


def evaluate_on_dataset(
    agent: PPOAgent,
    df,
    window_size: int,
    image_size: int,
    fee_rate: float,
    mdd_cap: float,
    volatility_lookback: int,
) -> Tuple[float, float]:
    env = TradingEnv(
        price_df=df,
        window_size=window_size,
        image_size=image_size,
        initial_balance=SIM_STARTING_BALANCE,
        fee_rate=fee_rate,
        mdd_cap=mdd_cap,
        volatility_lookback=volatility_lookback,
        reward_mdd_penalty=1.0,
    )
    obs = env.reset()
    done = False
    step_returns = []
    while not done:
        action, _, _ = agent.act(obs)
        obs, reward, done, info = env.step(action)
        step_returns.append(info.get("step_return", 0.0))
    steps_year = steps_per_year_from_timeframe(KLINE_INTERVAL)
    sharpe = annualized_sharpe(np.array(step_returns, dtype=np.float32), steps_year)
    mdd = compute_max_drawdown(np.array(env.equity_history, dtype=np.float32))
    return sharpe, mdd


def train_single_symbol() -> None:
    symbol = "BTC/USDT"
    timeframe = KLINE_INTERVAL
    data_path = os.path.join("data", f"{symbol.replace('/', '_')}_{timeframe}.csv")

    df = get_or_download_ohlcv(
        path=data_path,
        exchange_id=EXCHANGE_ID,
        symbol=symbol,
        timeframe=timeframe,
        since="2016-01-01",
        until="2025-09-30",
    )

    train_df, test_df = split_train_test(df, train_end="2022-12-31", test_start="2023-01-01")

    env = TradingEnv(
        price_df=train_df,
        window_size=64,
        image_size=32,
        initial_balance=SIM_STARTING_BALANCE,
        fee_rate=FEE_RATE,
        mdd_cap=0.10,
        volatility_lookback=20,
        reward_mdd_penalty=1.0,
    )

    obs = env.reset()
    obs_shape = obs.shape

    config = PPOConfig()
    agent = PPOAgent(obs_shape, env.action_space_n, config)

    max_steps = 200000
    steps_per_epoch = 2048
    total_steps = 0

    while total_steps < max_steps:
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []

        for _ in range(steps_per_epoch):
            action, logp, value = agent.act(obs)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            val_buf.append(value)

            next_obs, reward, done, info = env.step(action)
            rew_buf.append(reward)
            done_buf.append(1.0 if done else 0.0)
            obs = next_obs
            total_steps += 1

            if done:
                obs = env.reset()

        last_value = agent.value(obs)
        val_buf.append(last_value)

        rewards = np.array(rew_buf, dtype=np.float32)
        values = np.array(val_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)
        adv, ret = agent.compute_gae(rewards, values, dones)

        batch = {
            "obs": np.stack(obs_buf, axis=0),
            "act": np.array(act_buf, dtype=np.int64),
            "logp": np.array(logp_buf, dtype=np.float32),
            "adv": adv,
            "ret": ret,
        }

        info = agent.update(batch)

        train_sharpe, train_mdd = evaluate_on_dataset(
            agent,
            train_df,
            window_size=64,
            image_size=32,
            fee_rate=FEE_RATE,
            mdd_cap=0.10,
            volatility_lookback=20,
        )
        test_sharpe, test_mdd = evaluate_on_dataset(
            agent,
            test_df,
            window_size=64,
            image_size=32,
            fee_rate=FEE_RATE,
            mdd_cap=0.10,
            volatility_lookback=20,
        )

        print(
            f"steps={total_steps} "
            f"loss_pi={info.get('policy_loss', 0):.4f} "
            f"loss_v={info.get('value_loss', 0):.4f} "
            f"entropy={info.get('entropy', 0):.4f} "
            f"kl={info.get('approx_kl', 0):.5f} "
            f"train_sharpe={train_sharpe:.2f} train_mdd={train_mdd:.3f} "
            f"test_sharpe={test_sharpe:.2f} test_mdd={test_mdd:.3f}"
        )


if __name__ == "__main__":
    train_single_symbol()

import os
import sys
import time
import queue
import argparse
import multiprocessing as mp
from typing import Dict, Tuple

import numpy as np
import torch
import json

from config import EXCHANGE_ID, FEE_RATE, KLINE_INTERVAL, SIM_STARTING_BALANCE
from data_pipeline import get_or_download_ohlcv, split_train_test
from hardware_profiler import get_hardware_profile
from ppo_cnn_agent import PPOAgent, PPOConfig
from rl_env import TradingEnv
from risk_module import compute_max_drawdown


METRICS_FILE = "dfppo_metrics.json"


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


def serialize_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def load_state_dict_to_model(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state_dict)


def learner_process(
    update_queue: mp.Queue,
    weights_dict,
    train_df,
    test_df,
    window_size: int,
    image_size: int,
    config_dict: Dict,
) -> None:
    config = PPOConfig(**config_dict)
    env = TradingEnv(
        price_df=train_df,
        window_size=window_size,
        image_size=image_size,
        initial_balance=SIM_STARTING_BALANCE,
        fee_rate=FEE_RATE,
        mdd_cap=0.10,
        volatility_lookback=20,
        reward_mdd_penalty=1.0,
    )
    obs = env.reset()
    obs_shape = obs.shape

    agent = PPOAgent(obs_shape, env.action_space_n, config)
    version = 0
    weights_dict["state_dict"] = serialize_state_dict(agent.model)
    weights_dict["version"] = version
    weights_dict["stop"] = False

    steps_year = steps_per_year_from_timeframe(KLINE_INTERVAL)
    last_eval_time = time.time()
    eval_interval_sec = 300

    hp = get_hardware_profile()
    metrics_history = []
    last_update_info: Dict[str, float] = {}

    print("[Learner] Started. Waiting for gradient updates...")

    while True:
        if weights_dict.get("stop", False):
            break
        try:
            msg = update_queue.get(timeout=1.0)
        except queue.Empty:
            # Periodic evaluation
            now = time.time()
            if now - last_eval_time >= eval_interval_sec:
                train_sharpe, train_mdd = evaluate_on_dataset(
                    agent,
                    train_df,
                    window_size=window_size,
                    image_size=image_size,
                    fee_rate=FEE_RATE,
                    mdd_cap=0.10,
                    volatility_lookback=20,
                )
                test_sharpe, test_mdd = evaluate_on_dataset(
                    agent,
                    test_df,
                    window_size=window_size,
                    image_size=image_size,
                    fee_rate=FEE_RATE,
                    mdd_cap=0.10,
                    volatility_lookback=20,
                )

                metrics = {
                    "timestamp": now,
                    "version": int(version),
                    "train_sharpe": float(train_sharpe),
                    "train_mdd": float(train_mdd),
                    "test_sharpe": float(test_sharpe),
                    "test_mdd": float(test_mdd),
                    "compute_score": int(hp.compute_score),
                    "role": hp.role.value,
                }
                if last_update_info:
                    metrics.update(
                        {
                            "policy_loss": float(last_update_info.get("policy_loss", 0.0)),
                            "value_loss": float(last_update_info.get("value_loss", 0.0)),
                            "entropy": float(last_update_info.get("entropy", 0.0)),
                            "approx_kl": float(last_update_info.get("approx_kl", 0.0)),
                        }
                    )
                metrics_history.append(metrics)
                try:
                    with open(METRICS_FILE, "w") as f:
                        json.dump(metrics_history, f)
                except Exception:
                    pass

                print(
                    f"[Learner] eval: train_sharpe={train_sharpe:.2f} train_mdd={train_mdd:.3f} "
                    f"test_sharpe={test_sharpe:.2f} test_mdd={test_mdd:.3f}"
                )
                last_eval_time = now
            continue

        grads = msg.get("grads")
        if grads is None:
            continue

        info = msg.get("info") or {}
        last_update_info = info

        agent.optimizer.zero_grad()
        for p, g_np in zip(agent.model.parameters(), grads):
            if g_np is None:
                continue
            g = torch.from_numpy(g_np).to(agent.device)
            p.grad = g
        torch.nn.utils.clip_grad_norm_(agent.model.parameters(), config.max_grad_norm)
        agent.optimizer.step()

        version += 1
        if version % 5 == 0:
            weights_dict["state_dict"] = serialize_state_dict(agent.model)
            weights_dict["version"] = version
            print(f"[Learner] Applied update #{version} and broadcasted new weights.")


def actor_process(
    actor_id: int,
    weights_dict,
    update_queue: mp.Queue,
    train_df,
    window_size: int,
    image_size: int,
    config_dict: Dict,
) -> None:
    config = PPOConfig(**config_dict)
    env = TradingEnv(
        price_df=train_df,
        window_size=window_size,
        image_size=image_size,
        initial_balance=SIM_STARTING_BALANCE,
        fee_rate=FEE_RATE,
        mdd_cap=0.10,
        volatility_lookback=20,
        reward_mdd_penalty=1.0,
    )
    obs = env.reset()
    obs_shape = obs.shape

    agent = PPOAgent(obs_shape, env.action_space_n, config, device="cpu")

    local_version = -1
    steps_per_update = 512

    print(f"[Actor {actor_id}] Started.")

    while not weights_dict.get("stop", False):
        global_version = weights_dict.get("version", None)
        state_dict = weights_dict.get("state_dict", None)
        if global_version is not None and state_dict is not None and global_version != local_version:
            load_state_dict_to_model(agent.model, state_dict)
            local_version = global_version
            print(f"[Actor {actor_id}] Synced to weights version {local_version}.")

        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []

        for _ in range(steps_per_update):
            action, logp, value = agent.act(obs)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            val_buf.append(value)

            next_obs, reward, done, info = env.step(action)
            rew_buf.append(reward)
            done_buf.append(1.0 if done else 0.0)
            obs = next_obs

            if done:
                obs = env.reset()

        last_value = agent.value(obs)
        val_buf.append(last_value)

        rewards = np.array(rew_buf, dtype=np.float32)
        values = np.array(val_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)
        adv, ret = agent.compute_gae(rewards, values, dones)

        obs_arr = np.stack(obs_buf, axis=0)
        act_arr = np.array(act_buf, dtype=np.int64)
        logp_arr = np.array(logp_buf, dtype=np.float32)

        obs_t = torch.from_numpy(obs_arr).float()
        act_t = torch.from_numpy(act_arr).long()
        logp_old_t = torch.from_numpy(logp_arr).float()
        adv_t = torch.from_numpy(adv).float()
        ret_t = torch.from_numpy(ret).float()

        agent.model.train()
        agent.model.zero_grad()
        loss, info = agent._compute_loss(obs_t, act_t, logp_old_t, adv_t, ret_t)
        loss.backward()

        grads = []
        for p in agent.model.parameters():
            if p.grad is None:
                grads.append(None)
            else:
                grads.append(p.grad.detach().cpu().numpy())

        update_queue.put({"actor_id": actor_id, "grads": grads, "info": info})
        print(f"[Actor {actor_id}] Sent gradient update.")


def cleanup(processes, queue, weights_queue, stop_event):
    print("\nCleaning up processes...")
    stop_event.set()
    
    # First try to terminate processes gracefully
    for p in processes:
        if p.is_alive():
            try:
                p.terminate()
            except Exception as e:
                print(f"Error terminating process {p.pid}: {e}")
    
    # Wait for processes to terminate
    for i, p in enumerate(processes):
        try:
            p.join(timeout=2.0)
            if p.is_alive():
                print(f"Process {i} did not terminate, forcing...")
                p.kill()
                p.join(timeout=1.0)
        except Exception as e:
            print(f"Error joining process {i}: {e}")
    
    # Close queues
    try:
        queue.close()
        weights_queue.close()
    except Exception as e:
        print(f"Error closing queues: {e}")
    
    print("Cleanup complete.")


def validate_data():
    """Validate that we have all required data before starting training."""
    print("Validating data...")
    
    # Check for required symbols in config
    if not hasattr(config, 'TOKEN_WHITELIST') or not config.TOKEN_WHITELIST:
        raise ValueError("No symbols found in config.TOKEN_WHITELIST")
    
    # Check data directory exists
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Check we can load or download data for each symbol
    for symbol in config.TOKEN_WHITELIST:
        print(f"Checking data for {symbol}...")
        try:
            # This will raise an exception if data can't be loaded or downloaded
            data = get_or_download_ohlcv(
                path=os.path.join(data_dir, f"{symbol.replace('/', '_')}.csv"),
                exchange_id=config.EXCHANGE_ID,
                symbol=symbol,
                timeframe='1d',
                since='2016-01-01',
                until='2025-09-30'
            )
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
                
            print(f" {symbol}: {len(data)} data points")
            
        except Exception as e:
            print(f" Error with {symbol}: {str(e)}")
            raise
    
    print("Data validation complete. All required data is available.")


def run_dfppo_local() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-actors", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--skip-validation", action="store_true", 
                       help="Skip data validation (not recommended)")
    args = parser.parse_args()
    
    # Validate data before proceeding
    if not args.skip_validation:
        try:
            validate_data()
        except Exception as e:
            print("\nERROR: Data validation failed. Please fix the following issues:")
            print(f"- {str(e)}")
            print("\nCommon solutions:")
            print("1. Check your internet connection")
            print("2. Verify symbols in config.py are valid")
            print("3. Try running with --skip-validation (not recommended)")
            sys.exit(1)

    hp = get_hardware_profile()
    print(
        f"Hardware Profile: CS={hp.compute_score}, role={hp.role.value}, "
        f"CPU cores={hp.num_cpu_cores}, GPU={hp.gpu_name or 'None'} "
        f"VRAM={hp.total_vram_gb or 0}GB"
    )

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

    window_size = 64
    image_size = 32

    config = PPOConfig()
    config_dict = config.__dict__.copy()

    manager = mp.Manager()
    weights_dict = manager.dict()
    update_queue: mp.Queue = mp.Queue()

    learner = mp.Process(
        target=learner_process,
        args=(update_queue, weights_dict, train_df, test_df, window_size, image_size, config_dict),
    )
    learner.start()

    num_actors = max(1, (hp.num_cpu_cores or 2) - 1)
    actors = []
    for i in range(num_actors):
        p = mp.Process(
            target=actor_process,
            args=(i, weights_dict, update_queue, train_df, window_size, image_size, config_dict),
        )
        p.start()
        actors.append(p)

    print(f"Started {num_actors} actor processes.")

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Stopping DF-PPO training...")
    finally:
        weights_dict["stop"] = True
        for p in actors:
            p.join()
        learner.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_dfppo_local()

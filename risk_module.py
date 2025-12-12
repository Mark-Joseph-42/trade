import numpy as np


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    equity = np.asarray(equity_curve, dtype=np.float32).reshape(-1)
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / np.maximum(peaks, 1e-8)
    mdd = float(np.min(drawdowns))
    return abs(mdd)


def dynamic_position_size(
    equity: float,
    volatility: float,
    vol_target: float = 0.01,
    max_fraction: float = 0.10,
) -> float:
    if volatility <= 0.0:
        return max_fraction
    raw_fraction = vol_target / volatility
    fraction = max(0.0, min(max_fraction, raw_fraction))
    return float(fraction)


def kill_switch_triggered(equity_curve: np.ndarray, mdd_cap: float = 0.10) -> bool:
    mdd = compute_max_drawdown(equity_curve)
    return bool(mdd >= mdd_cap)

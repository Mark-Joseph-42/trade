import numpy as np
from typing import Iterable


def _resample_series(x: np.ndarray, target_length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = x.shape[0]
    if n == target_length:
        return x
    if n == 0:
        return np.zeros(target_length, dtype=np.float32)
    old_idx = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def _minmax_scale(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max == x_min:
        return np.zeros_like(x, dtype=np.float32)
    return (2.0 * (x - x_min) / (x_max - x_min) - 1.0).astype(np.float32)


def gaf_from_series(series: Iterable[float], image_size: int = 32, method: str = "summation") -> np.ndarray:
    x = np.asarray(list(series), dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros((image_size, image_size), dtype=np.float32)
    x = _resample_series(x, image_size)
    x = _minmax_scale(x)
    phi = np.arccos(np.clip(x, -1.0, 1.0))
    if method == "summation":
        gaf = np.cos(phi[:, None] + phi[None, :])
    elif method == "difference":
        gaf = np.sin(phi[:, None] - phi[None, :])
    else:
        raise ValueError(f"Unsupported GAF method: {method}")
    return gaf.astype(np.float32)


def gaf_from_multichannel(series_list: Iterable[Iterable[float]], image_size: int = 32, method: str = "summation") -> np.ndarray:
    channels = []
    for s in series_list:
        channels.append(gaf_from_series(s, image_size=image_size, method=method))
    if not channels:
        return np.zeros((0, image_size, image_size), dtype=np.float32)
    return np.stack(channels, axis=0)

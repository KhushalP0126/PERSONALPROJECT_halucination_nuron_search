import math

import numpy as np


def support_array(scores: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(scores, dtype=float)


def tail_count(scores: np.ndarray, fraction: float) -> int:
    return max(2, int(math.ceil(len(scores) * fraction)))


def full_conflict(scores: list[float] | np.ndarray) -> float:
    return float(np.std(support_array(scores)))


def late_conflict(scores: list[float] | np.ndarray, late_window: int) -> float:
    values = support_array(scores)
    late_values = values[-late_window:] if len(values) >= late_window else values
    return float(np.std(late_values))


def consensus_mean(scores: list[float] | np.ndarray) -> float:
    return float(np.mean(support_array(scores)))


def positive_layer_fraction(scores: list[float] | np.ndarray) -> float:
    values = support_array(scores)
    return float(np.mean(values > 0))


def sign_flip_count(scores: list[float] | np.ndarray) -> int:
    values = support_array(scores)
    signs = np.sign(values)
    nonzero_signs = signs[signs != 0]
    if len(nonzero_signs) <= 1:
        return 0
    return int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))


def late_slope(scores: list[float] | np.ndarray, late_window: int) -> float:
    values = support_array(scores)
    tail = values[-late_window:] if len(values) >= late_window else values
    if len(tail) <= 1:
        return 0.0
    x = np.arange(len(tail), dtype=float)
    return float(np.polyfit(x, tail, 1)[0])


def late_window_slope(scores: list[float] | np.ndarray, late_fraction: float) -> float:
    values = support_array(scores)
    count = tail_count(scores=values, fraction=late_fraction)
    tail = values[-count:]
    if len(tail) <= 1:
        return 0.0
    x = np.arange(len(tail), dtype=float)
    return float(np.polyfit(x, tail, 1)[0])


def mean_late_support(scores: list[float] | np.ndarray, late_fraction: float) -> float:
    values = support_array(scores)
    count = tail_count(scores=values, fraction=late_fraction)
    return float(np.mean(values[-count:]))

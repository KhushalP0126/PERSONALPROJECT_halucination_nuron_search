import math
import random

import numpy as np


def pooled_std(group_a: np.ndarray, group_b: np.ndarray) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    numerator = ((len(group_a) - 1) * group_a.var(ddof=1)) + ((len(group_b) - 1) * group_b.var(ddof=1))
    denominator = len(group_a) + len(group_b) - 2
    if denominator <= 0:
        return 0.0
    return float(math.sqrt(max(numerator / denominator, 0.0)))


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float | None:
    scale = pooled_std(group_a=group_a, group_b=group_b)
    if scale == 0:
        return None
    return float((group_a.mean() - group_b.mean()) / scale)


def common_language_effect_size(group_a: np.ndarray, group_b: np.ndarray) -> float | None:
    if len(group_a) == 0 or len(group_b) == 0:
        return None
    favorable = 0.0
    total = 0
    for left_value in group_a:
        for right_value in group_b:
            total += 1
            if left_value > right_value:
                favorable += 1.0
            elif left_value == right_value:
                favorable += 0.5
    if total == 0:
        return None
    return favorable / total


def permutation_p_value(
    group_a: np.ndarray,
    group_b: np.ndarray,
    permutations: int,
    rng: random.Random,
) -> float:
    observed = float(group_a.mean() - group_b.mean())
    combined = np.concatenate([group_a, group_b])
    left_count = len(group_a)
    extreme = 0
    for _ in range(permutations):
        indices = list(range(len(combined)))
        rng.shuffle(indices)
        shuffled = combined[indices]
        perm_left = shuffled[:left_count]
        perm_right = shuffled[left_count:]
        perm_diff = float(perm_left.mean() - perm_right.mean())
        if perm_diff >= observed:
            extreme += 1
    return (extreme + 1) / (permutations + 1)


def bootstrap_interval(
    group_a: np.ndarray,
    group_b: np.ndarray,
    bootstraps: int,
    rng: random.Random,
) -> tuple[float, float]:
    diffs: list[float] = []
    for _ in range(bootstraps):
        resampled_a = np.asarray([group_a[rng.randrange(len(group_a))] for _ in range(len(group_a))], dtype=float)
        resampled_b = np.asarray([group_b[rng.randrange(len(group_b))] for _ in range(len(group_b))], dtype=float)
        diffs.append(float(resampled_a.mean() - resampled_b.mean()))
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return float(lower), float(upper)


def binomial_p_value_greater_equal(successes: int, trials: int, baseline: float = 0.5) -> float:
    if trials <= 0:
        return 1.0
    probability = 0.0
    for count in range(successes, trials + 1):
        probability += math.comb(trials, count) * (baseline ** count) * ((1.0 - baseline) ** (trials - count))
    return probability


def midpoint_threshold(values: np.ndarray, labels: np.ndarray) -> float:
    correct_values = values[labels == 1]
    wrong_values = values[labels == 0]
    return float((correct_values.mean() + wrong_values.mean()) / 2.0)


def threshold_accuracy(values: np.ndarray, labels: np.ndarray, higher_is_correct: bool) -> tuple[float, float, int]:
    threshold = midpoint_threshold(values=values, labels=labels)
    if higher_is_correct:
        predictions = (values > threshold).astype(int)
    else:
        predictions = (values < threshold).astype(int)
    correct_count = int(np.sum(predictions == labels))
    return float(correct_count / len(labels)), threshold, correct_count


def roc_curve_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count == 0 or negative_count == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tpr = [0.0]
    fpr = [0.0]
    true_positives = 0
    false_positives = 0

    for index, label in enumerate(sorted_labels):
        if label == 1:
            true_positives += 1
        else:
            false_positives += 1

        next_score = sorted_scores[index + 1] if index + 1 < len(sorted_scores) else None
        if next_score != sorted_scores[index]:
            tpr.append(true_positives / positive_count)
            fpr.append(false_positives / negative_count)

    if tpr[-1] != 1.0 or fpr[-1] != 1.0:
        tpr.append(1.0)
        fpr.append(1.0)

    return np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    fpr, tpr = roc_curve_points(labels=labels, scores=scores)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])

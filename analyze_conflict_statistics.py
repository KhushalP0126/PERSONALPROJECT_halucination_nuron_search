import argparse
import json
import math
import random
from pathlib import Path

import numpy as np


DEFAULT_INPUT_FILE = Path("results/truthfulqa_consensus_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("results/conflict_statistics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether conflict separates correct from wrong benchmark answers."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the statistical summary will be written.",
    )
    parser.add_argument(
        "--score-field",
        default="support_scores",
        help="Field containing the per-layer support scores.",
    )
    parser.add_argument(
        "--label-field",
        default="label",
        help="Field containing the binary correctness label.",
    )
    parser.add_argument(
        "--late-window",
        type=int,
        default=5,
        help="Number of late layers to include in the late-conflict metric.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Number of permutation samples for the mean-difference test.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for the confidence interval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for permutation and bootstrap sampling.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Benchmark file must contain a non-empty JSON array.")
    return records


def labeled_records(records: list[dict], score_field: str, label_field: str) -> tuple[list[dict], int]:
    labeled: list[dict] = []
    skipped = 0
    for record in records:
        if record.get(label_field) not in {0, 1}:
            skipped += 1
            continue
        scores = record.get(score_field)
        if not isinstance(scores, list) or not scores:
            skipped += 1
            continue
        labeled.append(record)
    if not labeled:
        raise ValueError("No labeled records were available for statistical analysis.")
    return labeled, skipped


def full_conflict(scores: list[float]) -> float:
    return float(np.std(np.asarray(scores, dtype=float)))


def late_conflict(scores: list[float], late_window: int) -> float:
    values = np.asarray(scores, dtype=float)
    late_values = values[-late_window:] if len(values) >= late_window else values
    return float(np.std(late_values))


def threshold_accuracy(values: np.ndarray, labels: np.ndarray, lower_is_correct: bool = True) -> tuple[float, float, int]:
    correct_values = values[labels == 1]
    wrong_values = values[labels == 0]
    threshold = float((correct_values.mean() + wrong_values.mean()) / 2.0)
    if lower_is_correct:
        predictions = (values < threshold).astype(int)
    else:
        predictions = (values > threshold).astype(int)
    correct_count = int(np.sum(predictions == labels))
    return float(correct_count / len(labels)), threshold, correct_count


def pooled_std(correct_values: np.ndarray, wrong_values: np.ndarray) -> float:
    if len(correct_values) < 2 or len(wrong_values) < 2:
        return 0.0
    numerator = ((len(correct_values) - 1) * correct_values.var(ddof=1)) + (
        (len(wrong_values) - 1) * wrong_values.var(ddof=1)
    )
    denominator = len(correct_values) + len(wrong_values) - 2
    if denominator <= 0:
        return 0.0
    return float(math.sqrt(max(numerator / denominator, 0.0)))


def cohens_d(correct_values: np.ndarray, wrong_values: np.ndarray) -> float | None:
    scale = pooled_std(correct_values=correct_values, wrong_values=wrong_values)
    if scale == 0:
        return None
    return float((wrong_values.mean() - correct_values.mean()) / scale)


def common_language_effect_size(correct_values: np.ndarray, wrong_values: np.ndarray) -> float | None:
    if len(correct_values) == 0 or len(wrong_values) == 0:
        return None
    favorable = 0.0
    total = 0
    for wrong_value in wrong_values:
        for correct_value in correct_values:
            total += 1
            if wrong_value > correct_value:
                favorable += 1.0
            elif wrong_value == correct_value:
                favorable += 0.5
    if total == 0:
        return None
    return favorable / total


def permutation_p_value(
    correct_values: np.ndarray,
    wrong_values: np.ndarray,
    permutations: int,
    rng: random.Random,
) -> float:
    observed = float(wrong_values.mean() - correct_values.mean())
    combined = np.concatenate([correct_values, wrong_values])
    correct_count = len(correct_values)
    extreme = 0
    for _ in range(permutations):
        indices = list(range(len(combined)))
        rng.shuffle(indices)
        shuffled = combined[indices]
        perm_correct = shuffled[:correct_count]
        perm_wrong = shuffled[correct_count:]
        perm_diff = float(perm_wrong.mean() - perm_correct.mean())
        if perm_diff >= observed:
            extreme += 1
    return (extreme + 1) / (permutations + 1)


def bootstrap_interval(
    correct_values: np.ndarray,
    wrong_values: np.ndarray,
    bootstraps: int,
    rng: random.Random,
) -> tuple[float, float]:
    diffs: list[float] = []
    for _ in range(bootstraps):
        resampled_correct = np.asarray(
            [correct_values[rng.randrange(len(correct_values))] for _ in range(len(correct_values))],
            dtype=float,
        )
        resampled_wrong = np.asarray(
            [wrong_values[rng.randrange(len(wrong_values))] for _ in range(len(wrong_values))],
            dtype=float,
        )
        diffs.append(float(resampled_wrong.mean() - resampled_correct.mean()))
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return float(lower), float(upper)


def binomial_p_value_greater_equal(successes: int, trials: int, baseline: float = 0.5) -> float:
    if trials <= 0:
        return 1.0
    probability = 0.0
    for count in range(successes, trials + 1):
        probability += math.comb(trials, count) * (baseline ** count) * ((1.0 - baseline) ** (trials - count))
    return probability


def metric_summary(
    name: str,
    values: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    bootstraps: int,
    rng_seed: int,
) -> dict:
    correct_values = values[labels == 1]
    wrong_values = values[labels == 0]
    mean_diff = float(wrong_values.mean() - correct_values.mean())
    accuracy, threshold, threshold_correct = threshold_accuracy(values=values, labels=labels, lower_is_correct=True)
    rng_perm = random.Random(rng_seed)
    rng_boot = random.Random(rng_seed + 1)

    return {
        "metric_name": name,
        "sample_count": int(len(values)),
        "correct_count": int(len(correct_values)),
        "wrong_count": int(len(wrong_values)),
        "correct_mean": float(correct_values.mean()),
        "wrong_mean": float(wrong_values.mean()),
        "mean_difference_wrong_minus_correct": mean_diff,
        "cohens_d": cohens_d(correct_values=correct_values, wrong_values=wrong_values),
        "common_language_effect_size": common_language_effect_size(
            correct_values=correct_values,
            wrong_values=wrong_values,
        ),
        "permutation_p_value": permutation_p_value(
            correct_values=correct_values,
            wrong_values=wrong_values,
            permutations=permutations,
            rng=rng_perm,
        ),
        "bootstrap_95ci_wrong_minus_correct": bootstrap_interval(
            correct_values=correct_values,
            wrong_values=wrong_values,
            bootstraps=bootstraps,
            rng=rng_boot,
        ),
        "threshold": threshold,
        "threshold_accuracy": accuracy,
        "threshold_correct_count": threshold_correct,
        "threshold_binomial_p_value": binomial_p_value_greater_equal(
            successes=threshold_correct,
            trials=len(values),
        ),
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.out_dir)

    records = load_records(input_path)
    labeled, skipped = labeled_records(records=records, score_field=args.score_field, label_field=args.label_field)

    labels = np.asarray([int(record[args.label_field]) for record in labeled], dtype=int)
    conflict_values = np.asarray([full_conflict(record[args.score_field]) for record in labeled], dtype=float)
    late_conflict_values = np.asarray(
        [late_conflict(record[args.score_field], late_window=args.late_window) for record in labeled],
        dtype=float,
    )

    summary = {
        "input_path": str(input_path),
        "sample_count": len(records),
        "labeled_count": len(labeled),
        "skipped_unlabeled_count": skipped,
        "metrics": {
            "conflict": metric_summary(
                name="conflict",
                values=conflict_values,
                labels=labels,
                permutations=args.permutations,
                bootstraps=args.bootstraps,
                rng_seed=args.seed,
            ),
            "late_conflict": metric_summary(
                name="late_conflict",
                values=late_conflict_values,
                labels=labels,
                permutations=args.permutations,
                bootstraps=args.bootstraps,
                rng_seed=args.seed + 1000,
            ),
        },
    }

    save_json(summary, output_dir / "summary.json")

    conflict_summary = summary["metrics"]["conflict"]
    late_summary = summary["metrics"]["late_conflict"]
    print(
        f"labeled={len(labeled)} skipped_unlabeled={skipped} "
        f"conflict_accuracy={conflict_summary['threshold_accuracy']:.3f} "
        f"late_conflict_accuracy={late_summary['threshold_accuracy']:.3f}"
    )
    print(
        f"conflict_p={conflict_summary['permutation_p_value']:.4f} "
        f"late_conflict_p={late_summary['permutation_p_value']:.4f}"
    )
    print(f"Saved statistical summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

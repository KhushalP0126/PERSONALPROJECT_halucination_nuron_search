import argparse
from pathlib import Path
import random

import numpy as np

from detection.features import full_conflict, late_conflict
from detection.io import labeled_records, load_records, save_json
from detection.stats import (
    binomial_p_value_greater_equal,
    bootstrap_interval,
    cohens_d,
    common_language_effect_size,
    permutation_p_value,
    threshold_accuracy,
)


DEFAULT_INPUT_FILE = Path("results/truthfulqa_consensus_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("results/conflict_statistics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether conflict separates correct from wrong benchmark answers."
    )
    parser.add_argument("--in", dest="input_path", default=str(DEFAULT_INPUT_FILE), help="Path to the benchmark JSON file.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the statistical summary will be written.",
    )
    parser.add_argument("--score-field", default="support_scores", help="Field containing the per-layer support scores.")
    parser.add_argument("--label-field", default="label", help="Field containing the binary correctness label.")
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
    parser.add_argument("--seed", type=int, default=7, help="Random seed for permutation and bootstrap sampling.")
    return parser.parse_args()


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
    accuracy, threshold, threshold_correct = threshold_accuracy(values=values, labels=labels, higher_is_correct=False)

    return {
        "metric_name": name,
        "sample_count": int(len(values)),
        "correct_count": int(len(correct_values)),
        "wrong_count": int(len(wrong_values)),
        "correct_mean": float(correct_values.mean()),
        "wrong_mean": float(wrong_values.mean()),
        "mean_difference_wrong_minus_correct": float(wrong_values.mean() - correct_values.mean()),
        "cohens_d": cohens_d(group_a=wrong_values, group_b=correct_values),
        "common_language_effect_size": common_language_effect_size(group_a=wrong_values, group_b=correct_values),
        "permutation_p_value": permutation_p_value(
            group_a=wrong_values,
            group_b=correct_values,
            permutations=permutations,
            rng=random.Random(rng_seed),
        ),
        "bootstrap_95ci_wrong_minus_correct": bootstrap_interval(
            group_a=wrong_values,
            group_b=correct_values,
            bootstraps=bootstraps,
            rng=random.Random(rng_seed + 1),
        ),
        "threshold": threshold,
        "threshold_accuracy": accuracy,
        "threshold_correct_count": threshold_correct,
        "threshold_binomial_p_value": binomial_p_value_greater_equal(successes=threshold_correct, trials=len(values)),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    labeled, skipped = labeled_records(
        records=records,
        score_field=args.score_field,
        label_field=args.label_field,
        empty_error_message="No labeled records were available for statistical analysis.",
    )
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
        "late_window": args.late_window,
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
                rng_seed=args.seed + 100,
            ),
        },
    }

    save_json(summary, output_dir / "summary.json")

    conflict_summary = summary["metrics"]["conflict"]
    late_summary = summary["metrics"]["late_conflict"]
    print(
        f"labeled={len(labeled)} skipped_unlabeled={skipped} "
        f"conflict_acc={conflict_summary['threshold_accuracy']:.3f} "
        f"late_conflict_acc={late_summary['threshold_accuracy']:.3f} "
        f"conflict_p={conflict_summary['permutation_p_value']:.4f} "
        f"late_conflict_p={late_summary['permutation_p_value']:.4f}"
    )
    print(f"Saved conflict statistics to {output_dir}")


if __name__ == "__main__":
    main()

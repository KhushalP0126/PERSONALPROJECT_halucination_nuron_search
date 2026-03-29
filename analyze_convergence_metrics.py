import argparse
from pathlib import Path
import random

from detection.env import configure_matplotlib_env

configure_matplotlib_env(Path(__file__).resolve().parent)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from detection.features import late_slope
from detection.io import labeled_records, load_records, save_json
from detection.stats import (
    binomial_p_value_greater_equal,
    bootstrap_interval,
    cohens_d,
    common_language_effect_size,
    permutation_p_value,
    threshold_accuracy,
)


DEFAULT_INPUT_FILE = Path("results/truthfulqa_consensus_benchmark_reviewed.json")
DEFAULT_OUTPUT_DIR = Path("results/convergence_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze late-layer convergence signals as an alternative to the failed conflict hypothesis."
    )
    parser.add_argument("--in", dest="input_path", default=str(DEFAULT_INPUT_FILE), help="Path to the benchmark JSON file.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where convergence summaries and plots will be written.",
    )
    parser.add_argument("--score-field", default="support_scores", help="Field containing the truth-vs-false layer scores.")
    parser.add_argument(
        "--truth-model-field",
        default="truth_vs_model_scores",
        help="Field containing truth-vs-model layer scores.",
    )
    parser.add_argument("--label-field", default="label", help="Field containing binary labels where 1 means correct.")
    parser.add_argument("--late-window", type=int, default=5, help="How many final layers to use for the late-layer metrics.")
    parser.add_argument(
        "--early-window",
        type=int,
        default=5,
        help="How many initial layers to use for the early-to-late comparison.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Permutation samples for the mean-difference test.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=10000,
        help="Bootstrap resamples for confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed for resampling.")
    return parser.parse_args()


def extract_metric_values(
    records: list[dict],
    score_field: str,
    truth_model_field: str,
    early_window: int,
    late_window: int,
) -> dict[str, np.ndarray]:
    support_curves = [np.asarray(record[score_field], dtype=float) for record in records]
    truth_model_curves = [
        np.asarray(record.get(truth_model_field, record[score_field]), dtype=float) for record in records
    ]

    return {
        "late_slope": np.asarray([late_slope(curve, late_window=late_window) for curve in support_curves], dtype=float),
        "final_support": np.asarray([curve[-1] for curve in support_curves], dtype=float),
        "late_mean": np.asarray(
            [float(np.mean(curve[-late_window:] if len(curve) >= late_window else curve)) for curve in support_curves],
            dtype=float,
        ),
        "early_to_late_delta": np.asarray(
            [
                float(
                    np.mean(curve[-late_window:] if len(curve) >= late_window else curve)
                    - np.mean(curve[:early_window] if len(curve) >= early_window else curve)
                )
                for curve in support_curves
            ],
            dtype=float,
        ),
        "truth_model_final": np.asarray([curve[-1] for curve in truth_model_curves], dtype=float),
        "overall_abs_mean": np.asarray([float(np.mean(np.abs(curve))) for curve in support_curves], dtype=float),
    }


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
    accuracy, threshold, threshold_correct = threshold_accuracy(values=values, labels=labels, higher_is_correct=True)

    return {
        "metric_name": name,
        "sample_count": int(len(values)),
        "correct_count": int(len(correct_values)),
        "wrong_count": int(len(wrong_values)),
        "correct_mean": float(correct_values.mean()),
        "wrong_mean": float(wrong_values.mean()),
        "mean_difference_correct_minus_wrong": float(correct_values.mean() - wrong_values.mean()),
        "cohens_d": cohens_d(group_a=correct_values, group_b=wrong_values),
        "common_language_effect_size": common_language_effect_size(group_a=correct_values, group_b=wrong_values),
        "permutation_p_value": permutation_p_value(
            group_a=correct_values,
            group_b=wrong_values,
            permutations=permutations,
            rng=random.Random(rng_seed),
        ),
        "bootstrap_95ci_correct_minus_wrong": bootstrap_interval(
            group_a=correct_values,
            group_b=wrong_values,
            bootstraps=bootstraps,
            rng=random.Random(rng_seed + 1),
        ),
        "threshold": threshold,
        "threshold_accuracy": accuracy,
        "threshold_correct_count": threshold_correct,
        "threshold_binomial_p_value": binomial_p_value_greater_equal(successes=threshold_correct, trials=len(values)),
        "higher_is_correct": True,
    }


def plot_accuracy_bars(summary: dict, output_path: Path) -> None:
    metric_entries = list(summary["metrics"].values())
    labels = [entry["metric_name"] for entry in metric_entries]
    accuracies = [entry["threshold_accuracy"] for entry in metric_entries]

    plt.figure(figsize=(10, 5))
    colors = ["#4c72b0" if accuracy >= 0.6 else "#c44e52" for accuracy in accuracies]
    plt.bar(labels, accuracies, color=colors)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Threshold Accuracy")
    plt.title("Convergence Metrics on Reviewed TruthfulQA Benchmark")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_distribution(values: np.ndarray, labels: np.ndarray, metric_name: str, output_path: Path) -> None:
    correct_values = values[labels == 1]
    wrong_values = values[labels == 0]
    plt.figure(figsize=(9, 4))
    plt.hist(correct_values, bins=18, alpha=0.6, label="Correct", color="#4c72b0")
    plt.hist(wrong_values, bins=18, alpha=0.6, label="Wrong", color="#c44e52")
    plt.title(f"{metric_name} Distribution")
    plt.xlabel(metric_name)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


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
        empty_error_message="No labeled records were available for convergence analysis.",
    )
    labels = np.asarray([int(record[args.label_field]) for record in labeled], dtype=int)
    metric_values = extract_metric_values(
        records=labeled,
        score_field=args.score_field,
        truth_model_field=args.truth_model_field,
        early_window=args.early_window,
        late_window=args.late_window,
    )

    summary = {
        "input_path": str(input_path),
        "sample_count": len(records),
        "labeled_count": len(labeled),
        "skipped_unlabeled_count": skipped,
        "late_window": args.late_window,
        "early_window": args.early_window,
        "metrics": {},
    }

    for offset, (metric_name, values) in enumerate(metric_values.items()):
        summary["metrics"][metric_name] = metric_summary(
            name=metric_name,
            values=values,
            labels=labels,
            permutations=args.permutations,
            bootstraps=args.bootstraps,
            rng_seed=args.seed + offset * 100,
        )

    save_json(summary, output_dir / "summary.json")
    plot_accuracy_bars(summary=summary, output_path=output_dir / "metric_accuracy.png")
    for metric_name, values in metric_values.items():
        plot_distribution(
            values=values,
            labels=labels,
            metric_name=metric_name,
            output_path=output_dir / f"{metric_name}_distribution.png",
        )

    ranked = sorted(summary["metrics"].values(), key=lambda entry: entry["threshold_accuracy"], reverse=True)
    top_metric = ranked[0]
    print(
        f"labeled={len(labeled)} skipped_unlabeled={skipped} "
        f"best_metric={top_metric['metric_name']} accuracy={top_metric['threshold_accuracy']:.3f} "
        f"p={top_metric['permutation_p_value']:.4f}"
    )
    print(f"Saved convergence analysis to {output_dir}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import random

from detection.env import configure_matplotlib_env

configure_matplotlib_env(Path(__file__).resolve().parent)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from detection.features import late_slope, late_window_slope, mean_late_support
from detection.io import labeled_records, load_records, save_json, save_jsonl, with_record_indices
from detection.models import fit_logistic_regression, predict_probability, standardize_train_test
from detection.stats import (
    binomial_p_value_greater_equal,
    bootstrap_interval,
    cohens_d,
    common_language_effect_size,
    midpoint_threshold,
    pearson_correlation,
    permutation_p_value,
    roc_auc,
    roc_curve_points,
)


DEFAULT_INPUT_FILE = Path("results/truthfulqa_consensus_benchmark_reviewed.json")
DEFAULT_OUTPUT_DIR = Path("results/late_slope_holdout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a locked holdout test for minimal late-layer correctness signals."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=str(DEFAULT_INPUT_FILE),
        help="Development dataset path, or the full dataset to split if --holdout-in is omitted.",
    )
    parser.add_argument(
        "--holdout-in",
        dest="holdout_input_path",
        help="Optional separate holdout dataset path. If omitted, a stratified split is created from --in.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the holdout evaluation outputs will be written.",
    )
    parser.add_argument("--score-field", default="support_scores", help="Field containing the layer support scores.")
    parser.add_argument("--label-field", default="label", help="Field containing binary labels where 1 means correct.")
    parser.add_argument(
        "--late-window",
        type=int,
        default=5,
        help="Legacy fixed late window retained for compatibility in summaries.",
    )
    parser.add_argument(
        "--late-fraction",
        type=float,
        default=0.3,
        help="Fraction of final layers to use for late-window slope and mean-late-support variants.",
    )
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.7,
        help="Development fraction when splitting a single dataset into dev and holdout.",
    )
    parser.add_argument("--seed", type=int, default=23, help="Random seed for the split.")
    parser.add_argument("--permutations", type=int, default=10000, help="Permutation samples for the effect tests.")
    parser.add_argument("--bootstraps", type=int, default=10000, help="Bootstrap resamples for confidence intervals.")
    parser.add_argument("--steps", type=int, default=4000, help="Gradient steps for the 2-feature logistic regression.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for the 2-feature logistic regression.")
    parser.add_argument("--reg", type=float, default=0.01, help="L2 regularization for the 2-feature logistic regression.")
    parser.add_argument(
        "--error-analysis-count",
        type=int,
        default=20,
        help="How many holdout mistakes to save for manual analysis.",
    )
    return parser.parse_args()


def stratified_split_indices(labels: np.ndarray, dev_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    dev_indices: list[int] = []
    holdout_indices: list[int] = []

    for label_value in [1, 0]:
        label_indices = np.where(labels == label_value)[0]
        shuffled = np.array(label_indices, copy=True)
        rng.shuffle(shuffled)

        dev_count = max(1, int(round(len(shuffled) * dev_fraction)))
        dev_count = min(dev_count, len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
        dev_indices.extend(shuffled[:dev_count].tolist())
        holdout_indices.extend(shuffled[dev_count:].tolist())

    dev_indices.sort()
    holdout_indices.sort()
    return dev_indices, holdout_indices


def split_records(records: list[dict], label_field: str, dev_fraction: float, seed: int) -> tuple[list[dict], list[dict], dict]:
    labels = np.asarray([int(record[label_field]) for record in records], dtype=int)
    dev_indices, holdout_indices = stratified_split_indices(labels=labels, dev_fraction=dev_fraction, seed=seed)
    dev_records = [records[index] for index in dev_indices]
    holdout_records = [records[index] for index in holdout_indices]
    split_metadata = {
        "mode": "single_dataset_split",
        "dev_fraction": dev_fraction,
        "seed": seed,
        "dev_indices": [int(records[index]["_record_index"]) for index in dev_indices],
        "holdout_indices": [int(records[index]["_record_index"]) for index in holdout_indices],
    }
    return dev_records, holdout_records, split_metadata


def separate_datasets(dev_records: list[dict], holdout_records: list[dict]) -> dict:
    return {
        "mode": "separate_datasets",
        "holdout_is_fresh_dataset": True,
        "dev_indices": [int(record["_record_index"]) for record in dev_records],
        "holdout_indices": [int(record["_record_index"]) for record in holdout_records],
    }


def record_brief(records: list[dict]) -> list[dict]:
    return [
        {
            "record_index": int(record["_record_index"]),
            "label": int(record["label"]),
            "question": record.get("q"),
            "label_method": record.get("label_method"),
        }
        for record in records
    ]


def extract_feature_arrays(
    records: list[dict],
    score_field: str,
    late_window: int,
    late_fraction: float,
) -> dict[str, np.ndarray]:
    support_curves = [np.asarray(record[score_field], dtype=float) for record in records]
    return {
        "legacy_late_slope_fixed_window": np.asarray(
            [late_slope(curve, late_window=late_window) for curve in support_curves],
            dtype=float,
        ),
        "late_window_slope": np.asarray(
            [late_window_slope(curve, late_fraction=late_fraction) for curve in support_curves],
            dtype=float,
        ),
        "mean_late_support": np.asarray(
            [mean_late_support(curve, late_fraction=late_fraction) for curve in support_curves],
            dtype=float,
        ),
        "logit_confidence": np.asarray([float(record.get("logit_confidence", 0.0)) for record in records], dtype=float),
    }


def evaluated(values: np.ndarray, labels: np.ndarray, threshold: float, permutations: int, bootstraps: int, seed: int) -> dict:
    correct_values = values[labels == 1]
    wrong_values = values[labels == 0]
    predictions = (values > threshold).astype(int)
    correct_count = int(np.sum(predictions == labels))

    return {
        "sample_count": int(len(values)),
        "correct_count": int(np.sum(labels == 1)),
        "wrong_count": int(np.sum(labels == 0)),
        "correct_mean": float(correct_values.mean()),
        "wrong_mean": float(wrong_values.mean()),
        "mean_difference_correct_minus_wrong": float(correct_values.mean() - wrong_values.mean()),
        "cohens_d": cohens_d(group_a=correct_values, group_b=wrong_values),
        "common_language_effect_size": common_language_effect_size(group_a=correct_values, group_b=wrong_values),
        "permutation_p_value": permutation_p_value(
            group_a=correct_values,
            group_b=wrong_values,
            permutations=permutations,
            rng=random.Random(seed),
        ),
        "bootstrap_95ci_correct_minus_wrong": bootstrap_interval(
            group_a=correct_values,
            group_b=wrong_values,
            bootstraps=bootstraps,
            rng=random.Random(seed + 1),
        ),
        "roc_auc": roc_auc(labels=labels, scores=values),
        "threshold": float(threshold),
        "threshold_accuracy": float(correct_count / len(values)),
        "threshold_correct_count": correct_count,
        "threshold_binomial_p_value": binomial_p_value_greater_equal(successes=correct_count, trials=len(values)),
    }


def rank_choice(summary: dict) -> tuple[float, float]:
    auc = summary["development"]["roc_auc"]
    return (
        float(summary["development"]["threshold_accuracy"]),
        float(auc if auc is not None else float("-inf")),
    )


def prediction_rows(
    records: list[dict],
    labels: np.ndarray,
    feature_arrays: dict[str, np.ndarray],
    selected_variant_name: str,
    selected_threshold: float,
    combined_probabilities: np.ndarray,
    combined_threshold: float,
) -> list[dict]:
    rows: list[dict] = []
    selected_scores = feature_arrays[selected_variant_name]

    for index, record in enumerate(records):
        selected_score = float(selected_scores[index])
        combined_score = float(combined_probabilities[index])
        rows.append(
            {
                "record_index": int(record["_record_index"]),
                "question": record.get("q"),
                "label": int(labels[index]),
                "label_method": record.get("label_method"),
                "model_answer": record.get("model_answer"),
                "features": {
                    "legacy_late_slope_fixed_window": float(feature_arrays["legacy_late_slope_fixed_window"][index]),
                    "late_window_slope": float(feature_arrays["late_window_slope"][index]),
                    "mean_late_support": float(feature_arrays["mean_late_support"][index]),
                    "logit_confidence": float(feature_arrays["logit_confidence"][index]),
                },
                "selected_variant_name": selected_variant_name,
                "selected_variant_score": selected_score,
                "selected_variant_prediction": int(selected_score > selected_threshold),
                "combined_probability": combined_score,
                "combined_prediction": int(combined_score > combined_threshold),
            }
        )

    return rows


def error_analysis_rows(
    records: list[dict],
    labels: np.ndarray,
    feature_arrays: dict[str, np.ndarray],
    selected_variant_name: str,
    count: int,
) -> list[dict]:
    selected_scores = feature_arrays[selected_variant_name]
    high_score_wrong = [index for index in np.argsort(-selected_scores) if labels[index] == 0][: count // 2]
    low_score_correct = [index for index in np.argsort(selected_scores) if labels[index] == 1][: count - len(high_score_wrong)]

    rows: list[dict] = []
    for bucket_name, indices in [
        ("high_score_wrong", high_score_wrong),
        ("low_score_correct", low_score_correct),
    ]:
        for index in indices:
            record = records[index]
            rows.append(
                {
                    "bucket": bucket_name,
                    "record_index": int(record["_record_index"]),
                    "question": record.get("q"),
                    "label": int(labels[index]),
                    "label_method": record.get("label_method"),
                    "model_answer": record.get("model_answer"),
                    "selected_variant_name": selected_variant_name,
                    "selected_variant_score": float(selected_scores[index]),
                    "logit_confidence": float(feature_arrays["logit_confidence"][index]),
                }
            )
    return rows


def plot_distribution(
    dev_values: np.ndarray,
    dev_labels: np.ndarray,
    holdout_values: np.ndarray,
    holdout_labels: np.ndarray,
    threshold: float,
    feature_label: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for axis, values, labels, title in [
        (axes[0], dev_values, dev_labels, "Development"),
        (axes[1], holdout_values, holdout_labels, "Holdout"),
    ]:
        axis.hist(values[labels == 1], bins=14, alpha=0.6, label="Correct", color="#4c72b0")
        axis.hist(values[labels == 0], bins=14, alpha=0.6, label="Wrong", color="#c44e52")
        axis.axvline(threshold, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel(feature_label)

    axes[0].set_ylabel("Count")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_roc_curves(
    dev_labels: np.ndarray,
    dev_slope_scores: np.ndarray,
    dev_combined_scores: np.ndarray,
    holdout_labels: np.ndarray,
    holdout_slope_scores: np.ndarray,
    holdout_combined_scores: np.ndarray,
    slope_label: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

    panels = [
        ("Development", dev_labels, dev_slope_scores, dev_combined_scores),
        ("Holdout", holdout_labels, holdout_slope_scores, holdout_combined_scores),
    ]
    for axis, (title, labels, slope_scores, combined_scores) in zip(axes, panels, strict=True):
        slope_fpr, slope_tpr = roc_curve_points(labels=labels, scores=slope_scores)
        combined_fpr, combined_tpr = roc_curve_points(labels=labels, scores=combined_scores)
        slope_auc = roc_auc(labels=labels, scores=slope_scores)
        combined_auc = roc_auc(labels=labels, scores=combined_scores)

        axis.plot(slope_fpr, slope_tpr, label=f"{slope_label} (AUC={slope_auc:.3f})", color="#4c72b0")
        axis.plot(combined_fpr, combined_tpr, label=f"combined model (AUC={combined_auc:.3f})", color="#dd8452")
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#888888")
        axis.set_title(title)
        axis.set_xlabel("False Positive Rate")

    axes[0].set_ylabel("True Positive Rate")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_generalization_bars(
    selected_variant_name: str,
    selected_dev_accuracy: float,
    selected_holdout_accuracy: float,
    combined_dev_accuracy: float,
    combined_holdout_accuracy: float,
    output_path: Path,
) -> None:
    categories = ["Development", "Holdout"]
    selected_values = [selected_dev_accuracy, selected_holdout_accuracy]
    combined_values = [combined_dev_accuracy, combined_holdout_accuracy]
    x = np.arange(len(categories))
    width = 0.35

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(x - width / 2, selected_values, width=width, label=selected_variant_name, color="#4c72b0")
    axis.bar(x + width / 2, combined_values, width=width, label="combined model", color="#dd8452")
    axis.set_xticks(x, categories)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy")
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_source = with_record_indices(load_records(input_path))
    dev_labeled, dev_skipped = labeled_records(
        dev_source,
        score_field=args.score_field,
        label_field=args.label_field,
        empty_error_message="No labeled records were available for convergence analysis.",
    )

    if args.holdout_input_path:
        holdout_source = with_record_indices(load_records(Path(args.holdout_input_path)))
        holdout_labeled, holdout_skipped = labeled_records(
            holdout_source,
            score_field=args.score_field,
            label_field=args.label_field,
            empty_error_message="No labeled records were available for convergence analysis.",
        )
        dev_records = dev_labeled
        holdout_records = holdout_labeled
        split_metadata = separate_datasets(dev_records=dev_records, holdout_records=holdout_records)
        split_metadata["dev_input_path"] = str(input_path)
        split_metadata["holdout_input_path"] = str(Path(args.holdout_input_path))
        split_metadata["dev_skipped_unlabeled_count"] = dev_skipped
        split_metadata["holdout_skipped_unlabeled_count"] = holdout_skipped
    else:
        dev_records, holdout_records, split_metadata = split_records(
            records=dev_labeled,
            label_field=args.label_field,
            dev_fraction=args.dev_fraction,
            seed=args.seed,
        )
        split_metadata["input_path"] = str(input_path)
        split_metadata["skipped_unlabeled_count"] = dev_skipped
        split_metadata["holdout_is_fresh_dataset"] = False

    dev_labels = np.asarray([int(record[args.label_field]) for record in dev_records], dtype=int)
    holdout_labels = np.asarray([int(record[args.label_field]) for record in holdout_records], dtype=int)
    dev_features = extract_feature_arrays(
        records=dev_records,
        score_field=args.score_field,
        late_window=args.late_window,
        late_fraction=args.late_fraction,
    )
    holdout_features = extract_feature_arrays(
        records=holdout_records,
        score_field=args.score_field,
        late_window=args.late_window,
        late_fraction=args.late_fraction,
    )

    variant_names = ["late_window_slope", "mean_late_support"]
    variants: dict[str, dict] = {}
    for offset, variant_name in enumerate(variant_names, start=1):
        threshold = midpoint_threshold(values=dev_features[variant_name], labels=dev_labels)
        variants[variant_name] = {
            "development": evaluated(
                values=dev_features[variant_name],
                labels=dev_labels,
                threshold=threshold,
                permutations=args.permutations,
                bootstraps=args.bootstraps,
                seed=args.seed + offset,
            ),
            "holdout": evaluated(
                values=holdout_features[variant_name],
                labels=holdout_labels,
                threshold=threshold,
                permutations=args.permutations,
                bootstraps=args.bootstraps,
                seed=args.seed + 100 + offset,
            ),
            "locked_rule": {
                "metric": variant_name,
                "higher_is_more_truthful": True,
                "threshold_source": "midpoint between development correct and wrong means",
            },
        }

    selected_variant_name = max(variants.items(), key=lambda item: rank_choice(item[1]))[0]
    selected_variant_threshold = float(variants[selected_variant_name]["development"]["threshold"])

    train_x = np.column_stack([dev_features[selected_variant_name], dev_features["logit_confidence"]])
    holdout_x = np.column_stack([holdout_features[selected_variant_name], holdout_features["logit_confidence"]])
    train_x_scaled, holdout_x_scaled, feature_means, feature_stds = standardize_train_test(train_x=train_x, test_x=holdout_x)
    weights, bias = fit_logistic_regression(
        train_x=train_x_scaled,
        train_y=dev_labels.astype(float),
        steps=args.steps,
        learning_rate=args.lr,
        regularization=args.reg,
    )
    dev_combined_scores = predict_probability(features=train_x_scaled, weights=weights, bias=bias)
    holdout_combined_scores = predict_probability(features=holdout_x_scaled, weights=weights, bias=bias)
    combined_threshold = midpoint_threshold(values=dev_combined_scores, labels=dev_labels)

    combined_summary = {
        "feature_names": [selected_variant_name, "logit_confidence"],
        "feature_correlation": {
            "development_pearson": pearson_correlation(
                left=dev_features[selected_variant_name],
                right=dev_features["logit_confidence"],
            ),
            "holdout_pearson": pearson_correlation(
                left=holdout_features[selected_variant_name],
                right=holdout_features["logit_confidence"],
            ),
        },
        "training": {
            "steps": args.steps,
            "learning_rate": args.lr,
            "regularization": args.reg,
            "weights_on_standardized_features": {
                selected_variant_name: float(weights[0]),
                "logit_confidence": float(weights[1]),
            },
            "bias": float(bias),
            "feature_means": {
                selected_variant_name: float(feature_means[0]),
                "logit_confidence": float(feature_means[1]),
            },
            "feature_stds": {
                selected_variant_name: float(feature_stds[0]),
                "logit_confidence": float(feature_stds[1]),
            },
        },
        "development": evaluated(
            values=dev_combined_scores,
            labels=dev_labels,
            threshold=combined_threshold,
            permutations=args.permutations,
            bootstraps=args.bootstraps,
            seed=args.seed + 50,
        ),
        "holdout": evaluated(
            values=holdout_combined_scores,
            labels=holdout_labels,
            threshold=combined_threshold,
            permutations=args.permutations,
            bootstraps=args.bootstraps,
            seed=args.seed + 150,
        ),
        "locked_rule": {
            "metric": "two_feature_logistic_regression",
            "training_source": "development split only",
            "threshold_source": "midpoint between development correct and wrong predicted-probability means",
        },
    }

    dev_prediction_rows = prediction_rows(
        records=dev_records,
        labels=dev_labels,
        feature_arrays=dev_features,
        selected_variant_name=selected_variant_name,
        selected_threshold=selected_variant_threshold,
        combined_probabilities=dev_combined_scores,
        combined_threshold=combined_threshold,
    )
    holdout_prediction_rows = prediction_rows(
        records=holdout_records,
        labels=holdout_labels,
        feature_arrays=holdout_features,
        selected_variant_name=selected_variant_name,
        selected_threshold=selected_variant_threshold,
        combined_probabilities=holdout_combined_scores,
        combined_threshold=combined_threshold,
    )
    error_rows = error_analysis_rows(
        records=holdout_records,
        labels=holdout_labels,
        feature_arrays=holdout_features,
        selected_variant_name=selected_variant_name,
        count=args.error_analysis_count,
    )

    artifacts = {
        "summary": str(output_dir / "summary.json"),
        "development_predictions": str(output_dir / "development_predictions.jsonl"),
        "holdout_predictions": str(output_dir / "holdout_predictions.jsonl"),
        "error_analysis_candidates": str(output_dir / "error_analysis_candidates.json"),
        "distribution_plot": str(output_dir / "late_slope_holdout.png"),
        "roc_curve_plot": str(output_dir / "roc_curve.png"),
        "generalization_plot": str(output_dir / "generalization_accuracy.png"),
    }

    legacy_threshold = midpoint_threshold(values=dev_features["legacy_late_slope_fixed_window"], labels=dev_labels)
    summary = {
        "split": split_metadata,
        "legacy_baseline": {
            "metric_name": "legacy_late_slope_fixed_window",
            "late_window": args.late_window,
            "development": evaluated(
                values=dev_features["legacy_late_slope_fixed_window"],
                labels=dev_labels,
                threshold=legacy_threshold,
                permutations=args.permutations,
                bootstraps=args.bootstraps,
                seed=args.seed + 200,
            ),
            "holdout": evaluated(
                values=holdout_features["legacy_late_slope_fixed_window"],
                labels=holdout_labels,
                threshold=legacy_threshold,
                permutations=args.permutations,
                bootstraps=args.bootstraps,
                seed=args.seed + 300,
            ),
        },
        "candidate_variants": {
            "late_window_slope": {
                "definition": f"slope over the final {int(round(args.late_fraction * 100))}% of support layers",
                **variants["late_window_slope"],
            },
            "mean_late_support": {
                "definition": f"mean support over the final {int(round(args.late_fraction * 100))}% of support layers",
                **variants["mean_late_support"],
            },
        },
        "selected_variant": {
            "name": selected_variant_name,
            "selection_rule": "highest development threshold accuracy, breaking ties by development ROC AUC",
            "development_pearson_correlation_with_logit_confidence": pearson_correlation(
                left=dev_features[selected_variant_name],
                right=dev_features["logit_confidence"],
            ),
            "holdout_pearson_correlation_with_logit_confidence": pearson_correlation(
                left=holdout_features[selected_variant_name],
                right=holdout_features["logit_confidence"],
            ),
            **variants[selected_variant_name],
        },
        "combined_model": combined_summary,
        "final_setup": {
            "name": "combined_model",
            "family": "combined_model",
            "frozen_internal_feature": selected_variant_name,
            "frozen_classifier": "logistic_regression",
            "selection_rule": "lock the best single internal feature, then pair it with logit_confidence as the minimal practical system",
            "claim": (
                "Late-layer convergence provides a weak but generalizable internal signal of correctness, "
                "and improves prediction when combined with model confidence."
            ),
        },
        "development_records": record_brief(dev_records),
        "holdout_records": record_brief(holdout_records),
        "artifacts": artifacts,
        "caveat": (
            "If split.mode is single_dataset_split, this is only a post-hoc holdout estimate because the same "
            "reviewed dataset informed metric discovery. Using a separate reviewed holdout file is cleaner."
        ),
    }

    save_json(summary, output_dir / "summary.json")
    save_jsonl(dev_prediction_rows, output_dir / "development_predictions.jsonl")
    save_jsonl(holdout_prediction_rows, output_dir / "holdout_predictions.jsonl")
    save_json(error_rows, output_dir / "error_analysis_candidates.json")

    plot_distribution(
        dev_values=dev_features[selected_variant_name],
        dev_labels=dev_labels,
        holdout_values=holdout_features[selected_variant_name],
        holdout_labels=holdout_labels,
        threshold=selected_variant_threshold,
        feature_label=selected_variant_name,
        output_path=output_dir / "late_slope_holdout.png",
    )
    plot_roc_curves(
        dev_labels=dev_labels,
        dev_slope_scores=dev_features[selected_variant_name],
        dev_combined_scores=dev_combined_scores,
        holdout_labels=holdout_labels,
        holdout_slope_scores=holdout_features[selected_variant_name],
        holdout_combined_scores=holdout_combined_scores,
        slope_label=selected_variant_name,
        output_path=output_dir / "roc_curve.png",
    )
    plot_generalization_bars(
        selected_variant_name=selected_variant_name,
        selected_dev_accuracy=float(variants[selected_variant_name]["development"]["threshold_accuracy"]),
        selected_holdout_accuracy=float(variants[selected_variant_name]["holdout"]["threshold_accuracy"]),
        combined_dev_accuracy=float(combined_summary["development"]["threshold_accuracy"]),
        combined_holdout_accuracy=float(combined_summary["holdout"]["threshold_accuracy"]),
        output_path=output_dir / "generalization_accuracy.png",
    )

    print(
        f"selected_variant={selected_variant_name} "
        f"variant_dev_accuracy={variants[selected_variant_name]['development']['threshold_accuracy']:.3f} "
        f"variant_holdout_accuracy={variants[selected_variant_name]['holdout']['threshold_accuracy']:.3f} "
        f"combined_dev_accuracy={combined_summary['development']['threshold_accuracy']:.3f} "
        f"combined_holdout_accuracy={combined_summary['holdout']['threshold_accuracy']:.3f}"
    )
    print(f"Saved holdout evaluation to {output_dir}")


if __name__ == "__main__":
    main()

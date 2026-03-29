import argparse
from pathlib import Path

from detection.env import configure_matplotlib_env

configure_matplotlib_env(Path(__file__).resolve().parent)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from detection.features import sign_flip_count
from detection.io import labeled_records, load_records, save_json
from detection.models import (
    classification_metrics,
    fit_full_model,
    leave_one_out_predictions,
)


DEFAULT_INPUT_FILE = Path("results/truthfulqa_consensus_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("results/stability_detector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a simple internal-stability detector from layer-wise consensus scores."
    )
    parser.add_argument("--in", dest="input_path", default=str(DEFAULT_INPUT_FILE), help="Path to the benchmark JSON file.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where detector outputs will be written.",
    )
    parser.add_argument("--score-field", default="support_scores", help="Field containing the per-layer support scores.")
    parser.add_argument("--label-field", default="label", help="Field containing the binary correctness label.")
    parser.add_argument("--late-window", type=int, default=5, help="How many late layers to use for late-stage features.")
    parser.add_argument("--steps", type=int, default=4000, help="Gradient steps for logistic regression training.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for logistic regression training.")
    parser.add_argument("--reg", type=float, default=0.01, help="L2 regularization strength.")
    return parser.parse_args()


def extract_feature_dict(record: dict, score_field: str, late_window: int) -> dict[str, float]:
    scores = np.asarray(record[score_field], dtype=float)
    late_scores = scores[-late_window:] if len(scores) >= late_window else scores

    return {
        "overall_mean": float(np.mean(scores)),
        "conflict_std": float(np.std(scores)),
        "late_mean": float(np.mean(late_scores)),
        "late_conflict_std": float(np.std(late_scores)),
        "spread": float(np.max(scores) - np.min(scores)),
        "sign_flips": float(sign_flip_count(scores)),
        "logit_confidence": float(record.get("logit_confidence", 0.0)),
    }


def feature_matrix(records: list[dict], score_field: str, late_window: int) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    feature_names: list[str] | None = None
    metadata: list[dict] = []

    for record in records:
        feature_dict = extract_feature_dict(record=record, score_field=score_field, late_window=late_window)
        if feature_names is None:
            feature_names = list(feature_dict.keys())
        feature_rows.append([feature_dict[name] for name in feature_names])
        labels.append(int(record["label"]))
        metadata.append(
            {
                "question": record.get("q"),
                "label": int(record["label"]),
                "label_method": record.get("label_method"),
                "model_answer": record.get("model_answer"),
                "features": feature_dict,
            }
        )

    return np.asarray(feature_rows, dtype=float), np.asarray(labels, dtype=float), feature_names or [], metadata


def plot_feature_weights(feature_names: list[str], weights: np.ndarray, output_path: Path) -> None:
    order = np.argsort(weights)
    ordered_names = [feature_names[index] for index in order]
    ordered_weights = [weights[index] for index in order]

    plt.figure(figsize=(10, 5))
    colors = ["#b22222" if weight < 0 else "#2e8b57" for weight in ordered_weights]
    plt.barh(ordered_names, ordered_weights, color=colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Full-Data Logistic Feature Weights")
    plt.xlabel("Standardized coefficient")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    labeled, skipped_unlabeled = labeled_records(
        records=records,
        score_field=args.score_field,
        label_field=args.label_field,
        empty_error_message="No labeled records were available for detector training.",
    )
    features, labels, feature_names, metadata = feature_matrix(
        labeled,
        score_field=args.score_field,
        late_window=args.late_window,
    )

    probabilities, predictions = leave_one_out_predictions(
        features=features,
        labels=labels,
        steps=args.steps,
        learning_rate=args.lr,
        regularization=args.reg,
    )
    evaluation = classification_metrics(labels=labels, predictions=predictions)

    weights, bias, mean, std = fit_full_model(
        features=features,
        labels=labels,
        steps=args.steps,
        learning_rate=args.lr,
        regularization=args.reg,
    )

    prediction_rows = []
    for index, meta in enumerate(metadata):
        prediction_rows.append(
            {
                "question": meta["question"],
                "label": meta["label"],
                "prediction": int(predictions[index]),
                "probability_correct": float(probabilities[index]),
                "label_method": meta["label_method"],
                "model_answer": meta["model_answer"],
                "features": meta["features"],
            }
        )

    summary = {
        "input_path": str(input_path),
        "sample_count": len(records),
        "labeled_count": len(labeled),
        "skipped_unlabeled_count": skipped_unlabeled,
        "feature_names": feature_names,
        "evaluation": evaluation,
        "full_model": {
            "bias": float(bias),
            "weights": {feature_names[index]: float(weights[index]) for index in range(len(feature_names))},
            "standardization_mean": {feature_names[index]: float(mean[index]) for index in range(len(feature_names))},
            "standardization_std": {feature_names[index]: float(std[index]) for index in range(len(feature_names))},
        },
    }

    save_json(summary, output_dir / "summary.json")
    save_json({"predictions": prediction_rows}, output_dir / "predictions.json")
    plot_feature_weights(feature_names=feature_names, weights=weights, output_path=output_dir / "feature_weights.png")

    print(
        f"labeled={len(labeled)} skipped_unlabeled={skipped_unlabeled} "
        f"accuracy={evaluation['accuracy']:.3f} "
        f"balanced_accuracy={evaluation['balanced_accuracy']:.3f}"
    )
    print(f"Saved detector outputs to {output_dir}")


if __name__ == "__main__":
    main()

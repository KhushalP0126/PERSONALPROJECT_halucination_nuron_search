import argparse
import json
import os
from pathlib import Path

PROJECT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
PROJECT_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_DIR))

MATPLOTLIB_CACHE_DIR = PROJECT_CACHE_DIR / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_FILE = Path("results/truthfulqa_consensus_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("results/stability_detector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a simple internal-stability detector from layer-wise consensus scores."
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
        help="Directory where detector outputs will be written.",
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
        help="How many late layers to use for late-stage features.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4000,
        help="Gradient steps for logistic regression training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for logistic regression training.",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=0.01,
        help="L2 regularization strength.",
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
        raise ValueError("No labeled records were available for detector training.")
    return labeled, skipped


def sign_flip_count(scores: np.ndarray) -> int:
    signs = np.sign(scores)
    nonzero_signs = signs[signs != 0]
    if len(nonzero_signs) <= 1:
        return 0
    return int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))


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


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def class_weights(labels: np.ndarray) -> np.ndarray:
    positive_count = max(float(np.sum(labels == 1)), 1.0)
    negative_count = max(float(np.sum(labels == 0)), 1.0)
    total = len(labels)
    positive_weight = total / (2.0 * positive_count)
    negative_weight = total / (2.0 * negative_count)
    return np.where(labels == 1, positive_weight, negative_weight)


def fit_logistic_regression(
    train_x: np.ndarray,
    train_y: np.ndarray,
    steps: int,
    learning_rate: float,
    regularization: float,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(train_x.shape[1], dtype=float)
    bias = 0.0
    sample_weights = class_weights(train_y)

    for _ in range(steps):
        logits = train_x @ weights + bias
        probs = sigmoid(logits)
        errors = (probs - train_y) * sample_weights
        grad_w = (train_x.T @ errors) / len(train_x) + regularization * weights
        grad_b = float(np.mean(errors))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias


def predict_probability(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid(features @ weights + bias)


def confusion_counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((labels == 1) & (predictions == 1))),
        "tn": int(np.sum((labels == 0) & (predictions == 0))),
        "fp": int(np.sum((labels == 0) & (predictions == 1))),
        "fn": int(np.sum((labels == 1) & (predictions == 0))),
    }


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | None]:
    counts = confusion_counts(labels, predictions)
    accuracy = float(np.mean(labels == predictions))
    precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    specificity = safe_divide(counts["tn"], counts["tn"] + counts["fp"])
    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = (recall + specificity) / 2.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        **counts,
    }


def leave_one_out_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    steps: int,
    learning_rate: float,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros(len(labels), dtype=float)
    predictions = np.zeros(len(labels), dtype=int)

    for index in range(len(labels)):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[index] = False
        train_x = features[train_mask]
        train_y = labels[train_mask]
        test_x = features[index : index + 1]

        train_x_scaled, test_x_scaled, _, _ = standardize_train_test(train_x=train_x, test_x=test_x)
        weights, bias = fit_logistic_regression(
            train_x=train_x_scaled,
            train_y=train_y,
            steps=steps,
            learning_rate=learning_rate,
            regularization=regularization,
        )
        probability = float(predict_probability(test_x_scaled, weights, bias)[0])
        probabilities[index] = probability
        predictions[index] = int(probability >= 0.5)

    return probabilities, predictions


def fit_full_model(
    features: np.ndarray,
    labels: np.ndarray,
    steps: int,
    learning_rate: float,
    regularization: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    scaled_features, _, mean, std = standardize_train_test(train_x=features, test_x=features)
    weights, bias = fit_logistic_regression(
        train_x=scaled_features,
        train_y=labels,
        steps=steps,
        learning_rate=learning_rate,
        regularization=regularization,
    )
    return weights, bias, mean, std


def save_json(data: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, indent=2))


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
    labeled, skipped_unlabeled = labeled_records(records, score_field=args.score_field, label_field=args.label_field)
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
    evaluation = metrics(labels=labels, predictions=predictions)

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

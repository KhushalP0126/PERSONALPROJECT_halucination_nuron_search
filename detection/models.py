import numpy as np


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


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | None]:
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

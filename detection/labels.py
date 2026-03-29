def as_binary_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in {0, 1}:
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "correct", "truthful"}:
            return 1
        if normalized in {"0", "false", "wrong", "hallucinated"}:
            return 0
    raise ValueError(f"Expected a binary label, got: {value!r}")


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def is_unknown_target(text: str) -> bool:
    normalized = normalize_text(text)
    return normalized in {
        "unknown",
        "unknown_future",
        "i don't know",
        "i dont know",
        "cannot be known yet",
    }


def answer_indicates_unknown(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        phrase in normalized
        for phrase in (
            "i don't know",
            "i dont know",
            "cannot be known",
            "can't be known",
            "cannot know",
            "unknown",
        )
    )


def derive_binary_label(record: dict) -> int:
    answer = record.get("answer") or record.get("model_answer")
    truth = record.get("gt") or record.get("correct_answer")
    false_answer = record.get("incorrect_answer")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Could not derive a binary label because the record is missing an answer field.")
    if not isinstance(truth, str) or not truth.strip():
        raise ValueError("Could not derive a binary label because the record is missing a ground-truth field.")

    normalized_answer = normalize_text(answer)
    normalized_truth = normalize_text(truth)
    normalized_false = normalize_text(false_answer) if isinstance(false_answer, str) else ""

    if is_unknown_target(truth):
        return 1 if answer_indicates_unknown(answer) else 0
    if normalized_false and normalized_false in normalized_answer and normalized_truth not in normalized_answer:
        return 0
    if normalized_truth in normalized_answer:
        return 1
    if normalized_answer == normalized_truth:
        return 1
    return 0

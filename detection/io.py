import json
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Benchmark file must contain a non-empty JSON array.")
    return records


def save_json(payload: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def labeled_records(
    records: list[dict],
    score_field: str,
    label_field: str,
    empty_error_message: str,
) -> tuple[list[dict], int]:
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
        raise ValueError(empty_error_message)
    return labeled, skipped


def with_record_indices(records: list[dict]) -> list[dict]:
    enriched = []
    for index, record in enumerate(records):
        enriched_record = dict(record)
        enriched_record["_record_index"] = index
        enriched.append(enriched_record)
    return enriched

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_INPUT_FILE = Path("results/truthfulqa_consensus_benchmark.json")
DEFAULT_OUTPUT_FILE = Path("results/truthfulqa_consensus_benchmark_reviewed.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rapidly review ambiguous benchmark outputs and assign manual correctness labels."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Where to save the reviewed benchmark JSON file.",
    )
    parser.add_argument(
        "--review-all",
        action="store_true",
        help="Review every record instead of only ambiguous ones.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip records before this queue position.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of queued records to review in this session.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Print the queued records and exit without prompting for labels.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Benchmark file must contain a non-empty JSON array.")
    return records


def save_records(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


def resume_source(input_path: Path, output_path: Path) -> Path:
    if output_path.exists():
        return output_path
    return input_path


def review_queue(records: list[dict], review_all: bool) -> list[int]:
    queue: list[int] = []
    for index, record in enumerate(records):
        label = record.get("label")
        label_method = str(record.get("label_method", ""))
        if review_all or label is None or label_method.startswith("ambiguous"):
            queue.append(index)
    return queue


def summarize_record(index: int, record: dict) -> str:
    conflict = None
    scores = record.get("support_scores")
    if isinstance(scores, list) and scores:
        conflict = float(np.std(np.asarray(scores, dtype=float)))

    lines = [
        f"[{index}] {record.get('q', '<missing question>')}",
        f"label={record.get('label')} label_method={record.get('label_method')}",
    ]
    if conflict is not None:
        lines.append(f"conflict={conflict:.3f}")
    if "logit_confidence" in record:
        lines.append(f"logit_confidence={float(record['logit_confidence']):.3f}")

    lines.extend(
        [
            f"model_answer: {record.get('model_answer', '<missing>')}",
            "correct_refs:",
        ]
    )
    for text in (record.get("correct_answers") or [])[:3]:
        lines.append(f"  - {text}")
    lines.append("incorrect_refs:")
    for text in (record.get("incorrect_answers") or [])[:3]:
        lines.append(f"  - {text}")

    label_details = record.get("label_details")
    if isinstance(label_details, dict) and label_details:
        best_correct = label_details.get("best_correct_reference")
        best_correct_score = label_details.get("best_correct_score")
        best_incorrect = label_details.get("best_incorrect_reference")
        best_incorrect_score = label_details.get("best_incorrect_score")
        if best_correct or best_incorrect:
            lines.append(
                "best_match: "
                f"correct={best_correct_score!r} -> {best_correct!r}, "
                f"incorrect={best_incorrect_score!r} -> {best_incorrect!r}"
            )

    return "\n".join(lines)


def apply_manual_label(record: dict, manual_label: int) -> None:
    previous_label = record.get("label")
    previous_label_method = record.get("label_method")
    previous_label_details = record.get("label_details")

    label_details = dict(previous_label_details) if isinstance(previous_label_details, dict) else {}
    label_details["manual_review"] = {
        "previous_label": previous_label,
        "previous_label_method": previous_label_method,
        "previous_label_details": previous_label_details,
    }

    record["label"] = manual_label
    record["label_method"] = "manual_review_correct" if manual_label == 1 else "manual_review_incorrect"
    record["label_details"] = label_details


def print_queue_preview(records: list[dict], queue: list[int]) -> None:
    print(f"Queued {len(queue)} records for review.")
    for index in queue:
        print("-" * 80)
        print(summarize_record(index=index, record=records[index]))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    source_path = resume_source(input_path=input_path, output_path=output_path)
    records = load_records(source_path)
    queue = review_queue(records=records, review_all=args.review_all)
    queue = queue[args.start_index :]
    if args.limit is not None:
        queue = queue[: args.limit]

    if not queue:
        print("No queued records matched the review criteria.")
        return

    if args.preview_only:
        print_queue_preview(records=records, queue=queue)
        return

    print(
        f"Review source: {source_path}\n"
        f"Saving reviewed records to: {output_path}\n"
        f"Queued records: {len(queue)}"
    )

    reviewed_count = 0
    skipped_count = 0

    for offset, record_index in enumerate(queue, start=1):
        record = records[record_index]
        print("\n" + "=" * 80)
        print(f"Queue item {offset}/{len(queue)}")
        print(summarize_record(index=record_index, record=record))
        print("Enter: 1=correct, 0=wrong, s=skip, q=quit")

        while True:
            choice = input("> ").strip().lower()
            if choice in {"1", "0"}:
                apply_manual_label(record=record, manual_label=int(choice))
                reviewed_count += 1
                save_records(records=records, path=output_path)
                print(f"Saved manual label {choice} for record {record_index}.")
                break
            if choice == "s":
                skipped_count += 1
                print(f"Skipped record {record_index}.")
                break
            if choice == "q":
                save_records(records=records, path=output_path)
                print(
                    f"Stopped early. reviewed={reviewed_count} skipped={skipped_count} "
                    f"saved_to={output_path}"
                )
                return
            print("Use 1, 0, s, or q.")

    save_records(records=records, path=output_path)
    print(
        f"Finished review. reviewed={reviewed_count} skipped={skipped_count} "
        f"saved_to={output_path}"
    )


if __name__ == "__main__":
    main()

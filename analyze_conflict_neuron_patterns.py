import argparse
import json
import os
from collections import defaultdict
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

from analyze_neuron_contributions import (
    DEFAULT_MODEL,
    PHI2_MODEL,
    analyze_record_neurons,
    load_model,
    plot_contribution_histogram,
    plot_top_neurons,
    resolve_device,
    save_json,
    validate_device,
)


DEFAULT_INPUT_FILE = Path("results/truthfulqa_consensus_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("results/conflict_neuron_patterns")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare neuron patterns between high-conflict wrong answers and low-conflict correct answers."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to load. Use {PHI2_MODEL} if you want to try Phi-2.",
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
        help="Directory where the aggregate neuron analysis outputs will be written.",
    )
    parser.add_argument(
        "--high-k",
        type=int,
        default=3,
        help="How many high-conflict wrong examples to analyze.",
    )
    parser.add_argument(
        "--low-k",
        type=int,
        default=3,
        help="How many low-conflict correct examples to analyze.",
    )
    parser.add_argument(
        "--top-k-neurons",
        type=int,
        default=20,
        help="How many top supporting/opposing neurons to keep per analyzed case.",
    )
    parser.add_argument(
        "--layer-mode",
        choices=["auto", "sign_flip", "max_abs", "strongest_support", "strongest_opposition"],
        default="auto",
        help="How to choose the analysis layer when no explicit layer is provided.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Optional explicit layer to use for all selected records.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device to use. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--allow-remote-code",
        action="store_true",
        help="Allow custom model code when the selected model requires it.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Benchmark file must contain a non-empty JSON array.")
    return records


def labeled_records(records: list[dict]) -> list[tuple[int, dict]]:
    labeled: list[tuple[int, dict]] = []
    for index, record in enumerate(records):
        if record.get("label") in {0, 1} and isinstance(record.get("support_scores"), list):
            labeled.append((index, record))
    if not labeled:
        raise ValueError("No labeled records were available for conflict-neuron analysis.")
    return labeled


def conflict_value(record: dict) -> float:
    return float(np.std(record["support_scores"]))


def select_groups(labeled: list[tuple[int, dict]], high_k: int, low_k: int) -> tuple[list[tuple[int, dict]], list[tuple[int, dict]]]:
    wrong = [(index, record) for index, record in labeled if record["label"] == 0]
    correct = [(index, record) for index, record in labeled if record["label"] == 1]

    high_conflict_wrong = sorted(wrong, key=lambda item: conflict_value(item[1]), reverse=True)[:high_k]
    low_conflict_correct = sorted(correct, key=lambda item: conflict_value(item[1]))[:low_k]
    return high_conflict_wrong, low_conflict_correct


def aggregate_neuron_frequency(analyses: list[dict], field: str) -> list[dict]:
    counts: dict[int, dict] = defaultdict(lambda: {"count": 0, "net_contributions": []})
    for analysis in analyses:
        for entry in analysis.get(field, []):
            neuron_index = int(entry["neuron_index"])
            counts[neuron_index]["count"] += 1
            counts[neuron_index]["net_contributions"].append(float(entry["net_contribution"]))

    output = []
    for neuron_index, stats in counts.items():
        output.append(
            {
                "neuron_index": neuron_index,
                "count": stats["count"],
                "mean_net_contribution": float(np.mean(stats["net_contributions"])),
                "max_abs_net_contribution": float(np.max(np.abs(stats["net_contributions"]))),
            }
        )

    return sorted(
        output,
        key=lambda item: (item["count"], abs(item["mean_net_contribution"]), item["max_abs_net_contribution"]),
        reverse=True,
    )


def plot_frequency(entries: list[dict], title: str, output_path: Path, limit: int = 15) -> None:
    top_entries = entries[:limit]
    if not top_entries:
        return

    labels = [str(entry["neuron_index"]) for entry in reversed(top_entries)]
    counts = [entry["count"] for entry in reversed(top_entries)]
    plt.figure(figsize=(10, 5))
    plt.barh(labels, counts, color="#4c72b0")
    plt.title(title)
    plt.xlabel("Count across selected cases")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_case_outputs(output_dir: Path, analysis: dict, support_contrib, comparison_contrib) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_contribution_histogram(
        support_contrib=support_contrib,
        comparison_contrib=comparison_contrib,
        support_token=analysis["support_token"],
        comparison_token=analysis["comparison_token"],
        output_path=output_dir / "contribution_histogram.png",
    )
    plot_top_neurons(
        top_supporting=[(entry["neuron_index"], entry["net_contribution"]) for entry in analysis["top_supporting_neurons"]],
        top_opposing=[(entry["neuron_index"], entry["net_contribution"]) for entry in analysis["top_opposing_neurons"]],
        output_path=output_dir / "top_neurons.png",
    )
    save_json(analysis, output_dir / "neuron_contributions.json")


def analyze_group(
    name: str,
    selected_records: list[tuple[int, dict]],
    tokenizer,
    model,
    device: str,
    explicit_layer: int | None,
    layer_mode: str,
    top_k_neurons: int,
    output_root: Path,
) -> dict:
    analyses: list[dict] = []
    for record_index, record in selected_records:
        analysis, support_contrib, comparison_contrib, _ = analyze_record_neurons(
            record_index=record_index,
            record=record,
            tokenizer=tokenizer,
            model=model,
            device=device,
            explicit_layer=explicit_layer,
            layer_mode=layer_mode,
            top_k=top_k_neurons,
        )
        analysis["conflict"] = conflict_value(record)
        analysis["label"] = record["label"]
        analysis["label_method"] = record.get("label_method")
        analyses.append(analysis)

        case_dir = output_root / name / f"record_{record_index:03d}_layer_{analysis['layer_index']:02d}"
        save_case_outputs(case_dir, analysis, support_contrib=support_contrib, comparison_contrib=comparison_contrib)

    return {
        "group_name": name,
        "sample_count": len(analyses),
        "mean_conflict": float(np.mean([analysis["conflict"] for analysis in analyses])) if analyses else None,
        "selected_questions": [
            {
                "record_index": analysis["record_index"],
                "question": analysis["question"],
                "conflict": analysis["conflict"],
                "layer_index": analysis["layer_index"],
                "layer_selection_reason": analysis["layer_selection_reason"],
            }
            for analysis in analyses
        ],
        "common_supporting_neurons": aggregate_neuron_frequency(analyses, field="top_supporting_neurons"),
        "common_opposing_neurons": aggregate_neuron_frequency(analyses, field="top_opposing_neurons"),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device() if args.device == "auto" else args.device
    validate_device(device)

    input_path = Path(args.input_path)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    labeled = labeled_records(records)
    high_conflict_wrong, low_conflict_correct = select_groups(
        labeled=labeled,
        high_k=args.high_k,
        low_k=args.low_k,
    )

    print(f"Loading {args.model} on {device}...")
    tokenizer, model = load_model(
        model_name=args.model,
        device=device,
        allow_remote_code=args.allow_remote_code,
    )

    high_summary = analyze_group(
        name="high_conflict_wrong",
        selected_records=high_conflict_wrong,
        tokenizer=tokenizer,
        model=model,
        device=device,
        explicit_layer=args.layer,
        layer_mode=args.layer_mode,
        top_k_neurons=args.top_k_neurons,
        output_root=output_dir,
    )
    low_summary = analyze_group(
        name="low_conflict_correct",
        selected_records=low_conflict_correct,
        tokenizer=tokenizer,
        model=model,
        device=device,
        explicit_layer=args.layer,
        layer_mode=args.layer_mode,
        top_k_neurons=args.top_k_neurons,
        output_root=output_dir,
    )

    plot_frequency(
        high_summary["common_opposing_neurons"],
        title="High-Conflict Wrong: Recurring Opposing Neurons",
        output_path=output_dir / "high_conflict_wrong_opposing_frequency.png",
    )
    plot_frequency(
        low_summary["common_supporting_neurons"],
        title="Low-Conflict Correct: Recurring Supporting Neurons",
        output_path=output_dir / "low_conflict_correct_supporting_frequency.png",
    )

    summary = {
        "input_path": str(input_path),
        "high_conflict_wrong": high_summary,
        "low_conflict_correct": low_summary,
    }
    save_json(summary, output_dir / "summary.json")

    print(
        f"high_conflict_wrong={high_summary['sample_count']} "
        f"low_conflict_correct={low_summary['sample_count']}"
    )
    print(f"Saved conflict-neuron outputs to {output_dir}")


if __name__ == "__main__":
    main()

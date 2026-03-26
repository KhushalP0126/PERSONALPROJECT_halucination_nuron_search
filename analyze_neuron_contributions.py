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
import torch

from build_consensus_dataset import (
    DEFAULT_MODEL,
    PHI2_MODEL,
    apply_final_norm,
    load_model,
    resolve_device,
    validate_device,
)


DEFAULT_INPUT_FILE = Path("results/consensus_dataset.json")
DEFAULT_OUTPUT_DIR = Path("results/neuron_contributions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zoom into one consensus-analysis layer and estimate which neurons support or oppose the target token."
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
        help="Path to a consensus JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plots and JSON output will be written.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Which record in the input JSON to analyze.",
    )
    parser.add_argument(
        "--question-contains",
        help="Optional substring match for selecting a specific question.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Explicit transformer layer index to analyze. If omitted, the script picks one automatically.",
    )
    parser.add_argument(
        "--layer-mode",
        choices=["auto", "sign_flip", "max_abs", "strongest_support", "strongest_opposition"],
        default="auto",
        help="How to choose a layer when --layer is not provided.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many top supporting and opposing neurons to report.",
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
        raise FileNotFoundError(f"Consensus dataset not found: {path}")

    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Consensus dataset must be a non-empty JSON array.")

    return records


def select_record(records: list[dict], sample_index: int, question_contains: str | None) -> tuple[int, dict]:
    if question_contains:
        needle = question_contains.lower()
        for index, record in enumerate(records):
            question = str(record.get("q", ""))
            if needle in question.lower():
                return index, record
        raise ValueError(f"No record question contains {question_contains!r}.")

    bounded_index = max(0, min(sample_index, len(records) - 1))
    return bounded_index, records[bounded_index]


def raw_token_ids(text: str, tokenizer) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if token_ids:
        return token_ids

    token_ids = tokenizer.encode(f" {text}", add_special_tokens=False)
    if token_ids:
        return token_ids

    raise ValueError(f"Could not derive token ids from text: {text!r}")


def build_analysis_inputs(record: dict, tokenizer, device: str) -> dict[str, torch.Tensor]:
    prompt = record["analysis_prompt"]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if "shared_prefix_length" in record and "correct_answer" in record:
        shared_prefix_length = int(record["shared_prefix_length"])
        correct_ids = raw_token_ids(record["correct_answer"], tokenizer)
        analysis_ids = prompt_ids + correct_ids[:shared_prefix_length]
        input_ids = torch.tensor([analysis_ids], device=device)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    inputs = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}


def resolve_token_pair(record: dict, tokenizer) -> tuple[int, str, int, str, str]:
    if "correct_token_id" in record and "comparison_token_id" in record:
        support_id = int(record["correct_token_id"])
        comparison_id = int(record["comparison_token_id"])
        support_token = record.get("correct_token", tokenizer.decode([support_id]))
        comparison_token = record.get("comparison_token", tokenizer.decode([comparison_id]))
        return support_id, support_token, comparison_id, comparison_token, "record_support_pair"

    if "truth_token_id" in record and "false_token_id" in record:
        support_id = int(record["truth_token_id"])
        comparison_id = int(record["false_token_id"])
        support_token = record.get("truth_token", tokenizer.decode([support_id]))
        comparison_token = record.get("false_token", tokenizer.decode([comparison_id]))
        return support_id, support_token, comparison_id, comparison_token, "truth_vs_false_pair"

    raise ValueError("Could not infer the support/comparison token pair from the selected record.")


def auto_layer_from_sign_flip(scores: list[float]) -> int | None:
    candidates: list[tuple[float, float, int]] = []
    for index in range(1, len(scores)):
        previous = scores[index - 1]
        current = scores[index]
        if previous == 0 or current == 0 or previous * current < 0:
            candidates.append((abs(current - previous), abs(current), index))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][2]


def choose_layer(scores: list[float], explicit_layer: int | None, mode: str) -> tuple[int, str]:
    if explicit_layer is not None:
        if explicit_layer < 0 or explicit_layer >= len(scores):
            raise ValueError(f"Layer {explicit_layer} is outside the valid range 0..{len(scores) - 1}.")
        return explicit_layer, "explicit"

    if mode in {"auto", "sign_flip"}:
        sign_flip_layer = auto_layer_from_sign_flip(scores)
        if sign_flip_layer is not None:
            reason = "largest_sign_flip" if mode == "auto" else "sign_flip"
            return sign_flip_layer, reason
        if mode == "sign_flip":
            raise ValueError("No sign-flip layer exists for this record. Use --layer-mode auto or max_abs instead.")

    if mode == "strongest_support":
        return max(range(len(scores)), key=lambda index: scores[index]), "strongest_support"
    if mode == "strongest_opposition":
        return min(range(len(scores)), key=lambda index: scores[index]), "strongest_opposition"

    return max(range(len(scores)), key=lambda index: abs(scores[index])), "max_abs_support_score"


def forward_hidden_states(record: dict, tokenizer, model, device: str):
    inputs = build_analysis_inputs(record=record, tokenizer=tokenizer, device=device)
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    return outputs.hidden_states


def top_neurons(values: torch.Tensor, top_k: int, descending: bool) -> list[tuple[int, float]]:
    sorted_indices = torch.argsort(values, descending=descending)
    output: list[tuple[int, float]] = []
    for index in sorted_indices.tolist():
        value = float(values[index].item())
        if descending and value <= 0:
            continue
        if not descending and value >= 0:
            continue
        output.append((index, value))
        if len(output) >= top_k:
            break
    return output


def plot_contribution_histogram(
    support_contrib: torch.Tensor,
    comparison_contrib: torch.Tensor,
    support_token: str,
    comparison_token: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 4))
    plt.hist(support_contrib.cpu().numpy(), bins=60, alpha=0.6, label=f"Support token: {support_token!r}")
    plt.hist(comparison_contrib.cpu().numpy(), bins=60, alpha=0.6, label=f"Comparison token: {comparison_token!r}")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Neuron Contribution Distribution")
    plt.xlabel("Contribution")
    plt.ylabel("Neuron count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_top_neurons(
    top_supporting: list[tuple[int, float]],
    top_opposing: list[tuple[int, float]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    if top_supporting:
        support_indices = [str(index) for index, _ in reversed(top_supporting)]
        support_values = [value for _, value in reversed(top_supporting)]
        axes[0].barh(support_indices, support_values, color="#2e8b57")
    axes[0].set_title("Top Supporting Neurons")
    axes[0].set_xlabel("Net contribution")

    if top_opposing:
        oppose_indices = [str(index) for index, _ in top_opposing]
        oppose_values = [value for _, value in top_opposing]
        axes[1].barh(oppose_indices, oppose_values, color="#b22222")
    axes[1].set_title("Top Opposing Neurons")
    axes[1].set_xlabel("Net contribution")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_output_record(
    record_index: int,
    record: dict,
    layer_index: int,
    layer_reason: str,
    support_id: int,
    support_token: str,
    comparison_id: int,
    comparison_token: str,
    support_contrib: torch.Tensor,
    comparison_contrib: torch.Tensor,
    net_contrib: torch.Tensor,
    layer_support_score: float,
    reconstructed_support_score: float,
    bias_delta: float,
    top_k: int,
) -> dict:
    top_supporting = top_neurons(net_contrib, top_k=top_k, descending=True)
    top_opposing = top_neurons(net_contrib, top_k=top_k, descending=False)
    supporting_count = int((net_contrib > 0).sum().item())
    opposing_count = int((net_contrib < 0).sum().item())
    neutral_count = int((net_contrib == 0).sum().item())

    return {
        "record_index": record_index,
        "question": record.get("q"),
        "analysis_prompt": record.get("analysis_prompt"),
        "layer_index": layer_index,
        "layer_selection_reason": layer_reason,
        "support_token_id": support_id,
        "support_token": support_token,
        "comparison_token_id": comparison_id,
        "comparison_token": comparison_token,
        "pair_mode": record.get("comparison_mode") or record.get("model_comparison_mode") or "derived",
        "layer_support_score": layer_support_score,
        "reconstructed_support_score": reconstructed_support_score,
        "bias_delta": bias_delta,
        "net_effect": float(net_contrib.sum().item()),
        "supporting_neuron_count": supporting_count,
        "opposing_neuron_count": opposing_count,
        "neutral_neuron_count": neutral_count,
        "top_supporting_neurons": [
            {
                "neuron_index": index,
                "net_contribution": value,
                "support_contribution": float(support_contrib[index].item()),
                "comparison_contribution": float(comparison_contrib[index].item()),
            }
            for index, value in top_supporting
        ],
        "top_opposing_neurons": [
            {
                "neuron_index": index,
                "net_contribution": value,
                "support_contribution": float(support_contrib[index].item()),
                "comparison_contribution": float(comparison_contrib[index].item()),
            }
            for index, value in top_opposing
        ],
        "support_contributions": support_contrib.tolist(),
        "comparison_contributions": comparison_contrib.tolist(),
        "net_contributions": net_contrib.tolist(),
    }


def save_json(record: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(record, indent=2))


def main() -> None:
    args = parse_args()
    device = resolve_device() if args.device == "auto" else args.device
    validate_device(device)

    input_path = Path(args.input_path)
    records = load_records(input_path)
    record_index, record = select_record(
        records=records,
        sample_index=args.sample_index,
        question_contains=args.question_contains,
    )
    support_scores = record.get("support_scores")
    if not isinstance(support_scores, list) or not support_scores:
        raise ValueError("Selected record does not contain a non-empty support_scores field.")

    layer_index, layer_reason = choose_layer(
        scores=support_scores,
        explicit_layer=args.layer,
        mode=args.layer_mode,
    )

    print(f"Loading {args.model} on {device}...")
    tokenizer, model = load_model(
        model_name=args.model,
        device=device,
        allow_remote_code=args.allow_remote_code,
    )

    support_id, support_token, comparison_id, comparison_token, token_pair_mode = resolve_token_pair(
        record=record,
        tokenizer=tokenizer,
    )
    hidden_states = forward_hidden_states(record=record, tokenizer=tokenizer, model=model, device=device)
    if layer_index + 1 >= len(hidden_states):
        raise ValueError(
            f"Layer {layer_index} does not exist in the returned hidden states. "
            f"Hidden state count: {len(hidden_states)}"
        )

    selected_hidden = hidden_states[layer_index + 1]
    normalized_hidden = apply_final_norm(model=model, hidden_state=selected_hidden)[:, -1, :].squeeze(0)
    weight = model.lm_head.weight

    support_contrib = (normalized_hidden * weight[support_id]).detach().float().cpu()
    comparison_contrib = (normalized_hidden * weight[comparison_id]).detach().float().cpu()
    net_contrib = support_contrib - comparison_contrib

    bias = getattr(model.lm_head, "bias", None)
    bias_delta = 0.0
    if bias is not None:
        bias_delta = float((bias[support_id] - bias[comparison_id]).item())

    net_effect = float(net_contrib.sum().item())
    reconstructed_support_score = net_effect + bias_delta
    layer_support_score = float(support_scores[layer_index])

    output_dir = Path(args.out_dir) / f"record_{record_index:03d}_layer_{layer_index:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    histogram_path = output_dir / "contribution_histogram.png"
    plot_contribution_histogram(
        support_contrib=support_contrib,
        comparison_contrib=comparison_contrib,
        support_token=support_token,
        comparison_token=comparison_token,
        output_path=histogram_path,
    )

    top_neurons_path = output_dir / "top_neurons.png"
    plot_top_neurons(
        top_supporting=top_neurons(net_contrib, top_k=args.top_k, descending=True),
        top_opposing=top_neurons(net_contrib, top_k=args.top_k, descending=False),
        output_path=top_neurons_path,
    )

    output_record = build_output_record(
        record_index=record_index,
        record=record,
        layer_index=layer_index,
        layer_reason=layer_reason,
        support_id=support_id,
        support_token=support_token,
        comparison_id=comparison_id,
        comparison_token=comparison_token,
        support_contrib=support_contrib,
        comparison_contrib=comparison_contrib,
        net_contrib=net_contrib,
        layer_support_score=layer_support_score,
        reconstructed_support_score=reconstructed_support_score,
        bias_delta=bias_delta,
        top_k=args.top_k,
    )
    output_record["token_pair_mode"] = token_pair_mode

    summary_path = output_dir / "neuron_contributions.json"
    save_json(output_record, summary_path)

    print(
        f"record={record_index} layer={layer_index} layer_reason={layer_reason} "
        f"pair_mode={token_pair_mode} support_token={support_token!r} comparison_token={comparison_token!r}"
    )
    print(
        f"layer_support_score={layer_support_score:.6f} "
        f"reconstructed_support_score={reconstructed_support_score:.6f} "
        f"bias_delta={bias_delta:.6f}"
    )
    print(
        f"supporting_neurons={output_record['supporting_neuron_count']} "
        f"opposing_neurons={output_record['opposing_neuron_count']} "
        f"neutral_neurons={output_record['neutral_neuron_count']}"
    )
    print(f"Saved neuron attribution outputs to {output_dir}")


if __name__ == "__main__":
    main()

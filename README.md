# LLM Convergence Signals

This repository studies whether internal late-layer behavior can help predict when a local language model is correct versus hallucinating.

The project started from a broader "consensus" idea and is now organized around a simpler detection claim:

> Late-layer convergence provides complementary information to model confidence and improves correctness prediction.

## Why This Project Matters

Most hallucination work focuses on output text or external verification. This project asks a different question:

> Can you predict correctness from the model's own internal trajectory before adding a complex external system?

That makes it a useful project for:

- LLM evaluation
- interpretability-adjacent research engineering
- lightweight correctness prediction
- building minimal, reproducible ML systems instead of vague demos

## Current Locked Result

The current external test is run on a reviewed dev set and a separate reviewed holdout set.

- Legacy fixed-window `late_slope`: `0.580` holdout accuracy
- `late_window_slope` over the final 30% of layers: `0.620` holdout accuracy
- `late_window_slope + logit_confidence`: `0.640` holdout accuracy and `0.703` holdout ROC AUC

Interpretation:

- `late_window_slope` is the best single internal feature in the current setup
- `late_window_slope + logit_confidence` is the best minimal practical system
- the two features are nearly uncorrelated, so they appear complementary rather than redundant

This is a real, generalizing signal, but not a production-ready detector.

## What I Built

- A local pipeline for collecting layer-wise support signals from a small language model
- A TruthfulQA benchmarking workflow with manual review for ambiguous examples
- Multiple detection baselines, including conflict-based metrics, convergence metrics, and simple logistic models
- A locked dev/holdout evaluation pipeline with saved predictions, thresholds, plots, and error buckets
- A shared `detection/` package to keep the detection scripts organized and reusable

## What I Found

- The original fixed-window `late_slope` signal was weak but non-random
- Measuring the slope over the final 30% of layers works better than the original fixed-window version
- `late_window_slope` and `logit_confidence` are nearly uncorrelated, so they capture different information
- Combining them improves holdout performance beyond either simple baseline alone
- Failure cases suggest the internal signal tracks convergence or coherence more directly than ground truth

## Reproduce The Main Result

If the reviewed dev and holdout benchmark files already exist, the fastest way to reproduce the locked result is:

```bash
make external-test
```

Main artifact:

```text
results/late_slope_holdout/summary.json
```

## Key Figures

Distribution of the selected internal feature:

![Distribution Plot](results/late_slope_holdout/late_slope_holdout.png)

ROC comparison between the selected slope feature and the combined model:

![ROC Curve](results/late_slope_holdout/roc_curve.png)

Development vs holdout accuracy for the selected feature and the combined model:

![Generalization Accuracy](results/late_slope_holdout/generalization_accuracy.png)

## At A Glance

- Problem: detect correctness using internal model behavior
- Best single feature: `late_window_slope`
- Best minimal system: `late_window_slope + logit_confidence`
- Holdout result: `0.640` accuracy, `0.703` ROC AUC
- Takeaway: internal convergence is a useful but limited signal, and it works better when paired with confidence

## What I Would Do Next

- Run one robustness check on a slightly different prompt format or a fresh benchmark slice
- Test trajectory-shape features instead of only slope-based summaries
- Reduce ambiguity in edge-case labels so the signal is not partly capped by label noise

## Repository Structure

Core runtime:

- `hf_local.py`: shared local Hugging Face runtime
- `local_chat.py`: local prompt testing and interactive chat
- `check_mps.py`: Apple Silicon / MPS check

Data and benchmark construction:

- `build_scored_hidden_dataset.py`: collect hidden-state runs on seed prompts
- `build_consensus_dataset.py`: build per-layer support scores
- `prepare_truthfulqa_dataset.py`: build TruthfulQA paired datasets
- `benchmark_truthfulqa_consensus.py`: benchmark truthful vs false answer behavior

Detection and evaluation:

- `visualize_consensus_patterns.py`: early visualization and threshold checks
- `analyze_conflict_statistics.py`: test whether conflict separates correct from wrong answers
- `analyze_convergence_metrics.py`: exploratory late-layer metric sweep
- `train_stability_detector.py`: train a compact detector over support-curve features
- `evaluate_late_slope_holdout.py`: locked dev/holdout evaluation for the final minimal system

Label review and interpretation:

- `review_benchmark_labels.py`: manually review ambiguous examples
- `export_manual_review_csv.py`: export reviewed data for inspection
- `analyze_neuron_contributions.py`: neuron-level decomposition for one selected example
- `analyze_conflict_neuron_patterns.py`: compare recurring neuron patterns across groups

Shared detection utilities:

- `detection/io.py`: shared record loading and JSON helpers
- `detection/labels.py`: shared label normalization and fallback heuristics
- `detection/features.py`: shared conflict and convergence feature helpers
- `detection/stats.py`: shared statistics, thresholding, ROC/AUC, and correlations
- `detection/models.py`: shared logistic-regression helpers
- `detection/env.py`: shared matplotlib cache setup

## Setup

Create the environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check Apple Silicon acceleration:

```bash
python check_mps.py
```

Verified fast default runtime:

```python
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

Optional:

```python
model_name = "microsoft/phi-2"
```

## Quickstart

Single prompt:

```bash
python local_chat.py --prompt "What is the capital of France?" --temperature 0
```

Interactive loop:

```bash
python local_chat.py
```

Locked external evaluation:

```bash
make external-test
```

## Recommended Workflow

### 1. Build seed hidden-state data

```bash
python build_scored_hidden_dataset.py
```

Single question:

```bash
python build_scored_hidden_dataset.py --question "What is the capital of France?"
```

Output:

```text
results/scored_hidden_dataset.jsonl
```

### 2. Build the consensus dataset

```bash
python build_consensus_dataset.py
```

Output:

```text
results/consensus_dataset.json
```

Each record includes layer-wise support information such as:

- `support_scores`
- `positive_layer_fraction`
- `positive_layer_indices`
- `positive_layer_ranges`
- `strongest_support_layer`
- `strongest_opposition_layer`

Support is measured as:

```text
logit(correct_token) - logit(comparison_token)
```

### 3. Summarize support locations

```bash
python summarize_layer_support.py --in results/consensus_dataset.json
```

This shows where positive support tends to appear across layers.

### 4. Prepare TruthfulQA slices

Small slice:

```bash
python prepare_truthfulqa_dataset.py --limit 50
```

Repeatable dev and holdout slices:

```bash
python prepare_truthfulqa_dataset.py --limit 100 --offset 0 --shuffle-seed 17 --out data/truthfulqa_dev.json
python prepare_truthfulqa_dataset.py --limit 100 --offset 100 --shuffle-seed 17 --out data/truthfulqa_holdout.json
```

Output records look like:

```json
{
  "q": "...",
  "correct_answer": "...",
  "incorrect_answer": "...",
  "source": "truthfulqa_generation_validation"
}
```

### 5. Run the benchmark

```bash
python benchmark_truthfulqa_consensus.py --limit 50
```

Output:

```text
results/truthfulqa_consensus_benchmark.json
```

The benchmark output includes:

- `support_scores`
- `label`
- `label_method`
- `truth_vs_false_scores`
- `truth_vs_model_scores`
- `shared_prefix_text`
- `logit_confidence`

### 6. Review ambiguous labels

```bash
python review_benchmark_labels.py --in results/truthfulqa_consensus_benchmark.json
```

Default reviewed output:

```text
results/truthfulqa_consensus_benchmark_reviewed.json
```

Preview mode:

```bash
python review_benchmark_labels.py --preview-only --limit 5
```

### 7. Explore detection signals

Early visualization:

```bash
python visualize_consensus_patterns.py --in results/truthfulqa_consensus_benchmark.json
```

Conflict statistics:

```bash
python analyze_conflict_statistics.py --in results/truthfulqa_consensus_benchmark.json
```

Convergence metric sweep:

```bash
python analyze_convergence_metrics.py --in results/truthfulqa_consensus_benchmark_reviewed.json
```

Compact feature detector:

```bash
python train_stability_detector.py --in results/truthfulqa_consensus_benchmark.json
```

### 8. Run the locked external test

If you already have reviewed dev and holdout files:

```bash
python evaluate_late_slope_holdout.py \
  --in results/truthfulqa_dev_benchmark_reviewed.json \
  --holdout-in results/truthfulqa_holdout_benchmark_reviewed.json
```

Or:

```bash
make external-test
```

This evaluation:

- computes the legacy fixed-window `late_slope` baseline
- evaluates `late_window_slope` over the final 30% of layers
- evaluates `mean_late_support` over the final 30% of layers
- selects the better single-feature variant on development only
- trains a 2-feature logistic regression using the selected variant and `logit_confidence`
- evaluates the holdout set without retuning
- saves predictions, thresholds, correlations, and plots

## External Test Artifacts

The locked evaluation writes:

- `results/late_slope_holdout/summary.json`
- `results/late_slope_holdout/development_predictions.jsonl`
- `results/late_slope_holdout/holdout_predictions.jsonl`
- `results/late_slope_holdout/error_analysis_candidates.json`
- `results/late_slope_holdout/late_slope_holdout.png`
- `results/late_slope_holdout/roc_curve.png`
- `results/late_slope_holdout/generalization_accuracy.png`

These artifacts are the main reproducible output for the current claim.

## Makefile Shortcuts

Setup and runtime:

```bash
make setup
make check-mps
make ask QUESTION="What is the capital of France?"
```

Dataset and benchmark generation:

```bash
make hidden-dataset
make consensus
make layers
make truthfulqa LIMIT=20
make benchmark LIMIT=20
```

Detection and analysis:

```bash
make plots
make detector
make conflict-stats
make convergence
make holdout
make external-test
```

Review workflow:

```bash
make dev-benchmark
make test-benchmark
make dev-review
make test-review
make label-benchmark
```

Neuron inspection:

```bash
make neurons QUESTION_MATCH="capital of France" LAYER=19
make conflict-neurons
```

Useful overrides:

```bash
make ask MODEL="microsoft/phi-2" QUESTION="What is the capital of Germany?"
make plots INPUT=results/truthfulqa_consensus_benchmark.json OUTPUT_DIR=results/consensus_plots
make neurons INPUT=results/consensus_dataset.json QUESTION_MATCH="capital of France" LAYER=19 TOP_K=25
make conflict-neurons HIGH_K=2 LOW_K=2
make label-benchmark LIMIT=10
make conflict-stats STATS_INPUT=results/truthfulqa_consensus_benchmark_reviewed.json
make convergence CONVERGENCE_INPUT=results/truthfulqa_consensus_benchmark_reviewed.json
make holdout DEV_FRACTION=0.7
```

## Interpretation Guide

What the current results mean:

- the late-layer signal is real
- the exact metric definition matters
- `late_window_slope` is better than the original fixed-window `late_slope`
- combining the internal signal with `logit_confidence` improves prediction

What the current results do not mean:

- this is not state of the art
- this is not a production detector
- internal convergence is not the same thing as ground truth

Observed failure modes:

- high-score wrong cases often look like confident misconceptions or stereotype-style yes/no claims
- low-score correct cases are often weakly phrased, edge-case, or ambiguous-label examples

## Data Notes

Seed inputs:

- `data/seed_questions.jsonl`
- `data/consensus_seed_dataset.json`

Prepared benchmark pairs:

- `data/truthfulqa_pairs.json`
- `data/truthfulqa_dev.json`
- `data/truthfulqa_holdout.json`

Generated benchmark and result files under `results/` are mostly ignored by git.

## Limitations

- The consensus pipeline is a layer-level approximation, not a full causal attribution method.
- TruthfulQA evaluation is sensitive to ambiguous labeling, so reviewed files matter.
- Several scripts depend on benchmark files that are not generated by default unless you run the preparation workflow.

## Next Step

The next useful step after the locked result is a small robustness check on either:

- a slightly different prompt format
- a fresh benchmark slice

That would test whether the current signal survives minor distribution changes without reopening the whole feature search.

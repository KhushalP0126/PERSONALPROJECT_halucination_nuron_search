# Hallucination Consensus Analyzer (Local LLM - M2)

This project builds a local hallucination analysis pipeline and extends it into mechanistic interpretability by analyzing:

- which internal signals support an answer
- which signals oppose it
- whether the model reaches internal consensus or conflict

## Objective

Move from:

```text
black-box scoring
```

to:

```text
internal signal analysis (consensus vs opposition)
```

## Setup (Mac M2)

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verify MPS:

```python
import torch
print(torch.backends.mps.is_available())
```

You can also run:

```bash
python check_mps.py
```

## Makefile Shortcuts

You can run the main commands with `make` instead of typing the full Python command each time:

```bash
make setup
make check-mps
make prompt PROMPT="What is the capital of France?"
make day2
make day3
make analyze-layers
make truthfulqa-prepare
make truthfulqa-consensus
```

Optional overrides:

```bash
make prompt MODEL="microsoft/phi-2" PROMPT="What is the capital of Germany?"
make day3 DAY3_OUT=results/consensus_dataset_test.json
make truthfulqa-consensus TRUTHFULQA_LIMIT=20
```

## Model

Verified fast local path:

```python
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

Optional:

```python
model_name = "microsoft/phi-2"
```

Models are loaded through the shared runtime in `hf_local.py`. The project uses explicit device placement on `mps` and enables hidden-state extraction during analysis passes rather than relying on `device_map="mps"`.

## Core Pipeline

```text
question
   ↓
model forward pass
   ↓
hidden states (all layers)
   ↓
logit lens (per layer)
   ↓
support vs opposition scores
```

## Key Concept

Each layer contributes to the final prediction.

The Day 3 pipeline measures:

```text
support = logit(correct_token) - logit(comparison_token)
```

- positive: supports the correct answer
- negative: opposes the correct answer

When the model already predicts the correct token, the comparison token is the strongest alternative token instead of the correct token itself. That avoids collapsing the score to zero.

## Datasets

### Day 2 seed dataset

`data/day2_questions.jsonl`

Each entry contains:

```json
{
  "question": "...",
  "ground_truth": "...",
  "category": "..."
}
```

### Day 3 consensus dataset input

`data/base_dataset.json`

Each entry contains:

```json
{
  "q": "...",
  "a": "...",
  "label": "factual | reasoning | unanswerable"
}
```

### TruthfulQA benchmark input

Prepare it with:

```bash
python prepare_truthfulqa.py --limit 50
```

This creates:

```text
data/truthfulqa_balanced.json
```

Each entry contains:

```json
{
  "q": "...",
  "correct_answer": "...",
  "incorrect_answer": "...",
  "source": "truthfulqa_generation_validation"
}
```

## Implementation

### 1. Day 1 local chat

Single prompt:

```bash
python local_chat.py --prompt "What is the capital of France?" --temperature 0
```

Interactive loop:

```bash
python local_chat.py
```

### 2. Day 2 answer -> score -> hidden pipeline

Run on the labeled seed set:

```bash
python day2_pipeline.py
```

This produces records with:

- `question`
- `answer`
- `score`
- `score_method`
- `score_evidence`
- `hidden`
- `hidden_shape`

Output file:

```text
results/day2_results.jsonl
```

### 3. Day 3 consensus pipeline

Run the consensus dataset builder:

```bash
python day3_consensus.py
```

This:

- runs a forward pass with `output_hidden_states=True`
- projects each layer through final norm plus `lm_head`
- extracts the next-token distribution at each layer
- compares the target token against the model prediction or strongest alternative

Output file:

```text
results/consensus_dataset.json
```

### 4. TruthfulQA consensus benchmark

Prepare a small benchmark slice:

```bash
python prepare_truthfulqa.py --limit 50
```

Run the paired truth-vs-false analysis:

```bash
python truthfulqa_consensus.py --limit 50
```

Or with `make`:

```bash
make truthfulqa-prepare TRUTHFULQA_LIMIT=50
make truthfulqa-consensus TRUTHFULQA_LIMIT=50
```

This path compares the token for the truthful answer against the token for a common false answer from TruthfulQA, which is a stronger benchmark signal than only comparing against the model's own predicted token.

When the truthful and false answers share the same opening tokens, the benchmark automatically moves to the first token where they diverge. For example, if both answers start with `Fortune cookies originated in ...`, the analysis compares `San` versus `China` rather than comparing the shared prefix.

## Core Mechanics

### Forward pass with hidden states

```python
outputs = model(
    **inputs,
    output_hidden_states=True,
    return_dict=True,
    use_cache=False,
)
```

### Logit lens per layer

```python
for hidden_state in hidden_states[1:]:
    normalized_hidden = apply_final_norm(model, hidden_state)
    logits = model.lm_head(normalized_hidden[:, -1, :])
```

### Consensus / opposition scoring

```python
correct_score = layer_logit[0, correct_id].item()
comparison_score = layer_logit[0, comparison_id].item()
support_score = correct_score - comparison_score
```

## Output Format

Each Day 3 sample looks like:

```json
{
  "q": "What is the capital of France?",
  "gt": "Paris",
  "label": "factual",
  "answer": "Paris is the capital of France.",
  "predicted_token": "Paris",
  "comparison_token": "The",
  "comparison_mode": "top_alternative",
  "correct_token": "Paris",
  "support_scores": [0.1, -0.4, 0.8],
  "consensus_mean": -0.2,
  "positive_layer_fraction": 0.33
}
```

## Interpretation

- mostly positive scores: internal agreement toward the target answer
- mixed scores: conflict or unstable convergence
- mostly negative scores: internal opposition or hallucination tendency

## What This Reveals

Instead of asking:

> Is the answer wrong?

you can ask:

> Did the model internally agree with itself?

## Next Steps

### Day 4

- correlate support patterns against labels
- test whether hallucinations show stronger conflict

### Day 5

- train a classifier on `support_scores`
- predict hallucination risk from internal signals

### Day 6+

- visualize layer-by-layer agreement
- detect unstable answers before output
- test causal interventions

## Notes

- this is an approximation, not exact neuron attribution
- signals are distributed across layers, not isolated to one unit
- interpretability remains an active research area
- the current Day 3 probe uses first-token consensus because it is the cleanest stable signal for a first pass

## Key Insight

Hallucination is not just a single wrong output.

It can be framed as:

```text
failure of internal consensus formation
```

## Stack

- Hugging Face Transformers
- PyTorch with MPS on Apple Silicon
- local causal language models such as TinyLlama and Phi-2

## Future Ideas

- activation patching
- feature direction analysis
- real-time hallucination warnings

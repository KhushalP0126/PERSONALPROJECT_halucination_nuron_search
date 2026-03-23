# Local Hugging Face Chat on Apple Silicon

This project gives you a minimal local text-generation setup for an Apple Silicon Mac.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Verify MPS

```bash
python check_mps.py
```

Expected output:

```text
True
```

## Run The Verified Fast Path

The default model is TinyLlama because it loads faster and was verified to generate correctly on Apple Metal.

Single prompt:

```bash
python local_chat.py --prompt "What is the capital of France?" --temperature 0
```

Interactive loop:

```bash
python local_chat.py
```

## Try Phi-2

If you want the original Phi-2 path from your plan:

```bash
python local_chat.py --model microsoft/phi-2 --prompt "What is the capital of France?" --temperature 0
```

## If Generation Is Slow

Reduce output length:

```bash
python local_chat.py --max-new-tokens 50
```

For more stable factual answers, prefer:

```bash
python local_chat.py --temperature 0
```

## Sanity Checks

Try:

- `Capital of Germany?`
- `Why is the sky blue?`
- `Who won the 2030 World Cup?`

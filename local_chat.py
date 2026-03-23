import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PHI2_MODEL = "microsoft/phi-2"


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    if device in {"mps", "cuda"}:
        return torch.float16
    return torch.float32


def should_trust_remote_code(model_name: str, allow_remote_code: bool) -> bool:
    if allow_remote_code:
        return True
    return model_name.lower() == PHI2_MODEL


def validate_device(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but PyTorch cannot access it in this session.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")


def load_model(
    model_name: str,
    device: str,
    allow_remote_code: bool,
):
    trust_remote_code = should_trust_remote_code(model_name, allow_remote_code)
    dtype = resolve_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    model.to(device)
    model.eval()

    if getattr(model.generation_config, "max_length", None) is not None:
        model.generation_config.max_length = None

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def tokenize_prompt(prompt: str, tokenizer, device: str):
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    return {key: value.to(device) for key, value in inputs.items()}


def generate(
    prompt: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenize_prompt(prompt=prompt, tokenizer=tokenizer, device=device)

    do_sample = temperature > 0
    generation_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def interactive_loop(
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> None:
    print("Interactive mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        answer = generate(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print(answer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Hugging Face text-generation model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to load. Use {PHI2_MODEL} if you want to try Phi-2.",
    )
    parser.add_argument(
        "--prompt",
        help="Generate a single response instead of starting the interactive loop.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Set to 0 for greedy decoding.",
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


def main() -> None:
    args = parse_args()
    device = resolve_device() if args.device == "auto" else args.device
    validate_device(device)

    print(f"Loading {args.model} on {device}...")
    tokenizer, model = load_model(
        model_name=args.model,
        device=device,
        allow_remote_code=args.allow_remote_code,
    )

    if args.prompt:
        answer = generate(
            prompt=args.prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(answer)
        return

    interactive_loop(
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

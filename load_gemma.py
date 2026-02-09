#!/usr/bin/env python3
"""Load google/gemma-2-2b from Hugging Face."""

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "google/gemma-2-2b"

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Successfully loaded {str(tokenizer)[:100]}")

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    )

    print(f"Successfully loaded {model_name}")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")


if __name__ == "__main__":
    main()

"""Split a base text file into consecutive chunks of n tokens.

Usage:
    python generate_prompt.py [--config CONFIG_PATH]

Reads config.yaml (or a custom path) for chunk sizes, encoding, and I/O paths.
Outputs files like data/prompt1_32.txt, prompt2_32.txt, ... that concatenate
back to the original text (minus trailing tokens that don't fill a full chunk).
"""

import argparse
from pathlib import Path

import tiktoken
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def chunk_tokens(tokens: list[int], n: int) -> list[list[int]]:
    """Split token list into consecutive chunks of exactly n tokens."""
    return [tokens[i : i + n] for i in range(0, len(tokens) - n + 1, n)]


def main():
    parser = argparse.ArgumentParser(description="Generate fixed-length token chunks from a base text.")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    prompts_dir = args.config.parent
    base_text_path = prompts_dir / config["base_text"]
    output_dir = prompts_dir / config["output_dir"]
    encoding_name = config.get("encoding", "cl100k_base")
    chunk_sizes = config["chunk_sizes"]

    enc = tiktoken.get_encoding(encoding_name)
    text = base_text_path.read_text(encoding="utf-8")
    tokens = enc.encode(text)

    print(f"Base text: {len(tokens)} tokens ({encoding_name})")

    for n in chunk_sizes:
        chunks = chunk_tokens(tokens, n)
        size_dir = output_dir
        size_dir.mkdir(parents=True, exist_ok=True)

        for i, chunk in enumerate(chunks, start=1):
            out_path = size_dir / f"prompt{i}_{n}.txt"
            out_path.write_text(enc.decode(chunk), encoding="utf-8")

        leftover = len(tokens) % n
        print(f"  n={n}: {len(chunks)} chunks written to {size_dir}/ ({leftover} trailing tokens dropped)")


if __name__ == "__main__":
    main()

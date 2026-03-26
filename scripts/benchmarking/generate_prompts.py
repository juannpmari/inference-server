#!/usr/bin/env python3
"""
Generate benchmark input prompt pools for the two input lengths defined in
docs/benchmarking/plan.md.

Input pools:
  short : ~32 input tokens
  long  : ~64 input tokens

The benchmark runner combines these with max_tokens (32 or 64) at request
time to form the four workloads (short-short, short-long, long-short,
long-long).

Usage:
    python scripts/benchmarking/generate_prompts.py \
        [--model Qwen/Qwen2-0.5B-Instruct] \
        [--output-dir benchmarks/prompts] \
        [--prompts-per-pool 25] \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

# ── input pool definitions ──────────────────────────────────────────
INPUT_POOLS: dict[str, int] = {
    "short": 32,
    "long": 64,
}

# ── diverse seed topics for prompt variety ──────────────────────────
SEED_TOPICS = [
    "quantum computing applications in drug discovery",
    "the history of the Silk Road trade routes",
    "how photosynthesis works in desert plants",
    "the economics of renewable energy storage",
    "neural network architectures for image recognition",
    "the cultural significance of fermented foods",
    "space debris mitigation strategies for low earth orbit",
    "the mathematics behind public key cryptography",
    "biodiversity loss in tropical rainforests",
    "the physics of superconducting materials",
    "machine learning for weather prediction models",
    "ancient Roman engineering and aqueduct design",
    "the psychology of decision making under uncertainty",
    "ocean current patterns and their effect on climate",
    "advances in CRISPR gene editing technology",
    "the philosophy of consciousness and artificial minds",
    "sustainable urban planning in megacities",
    "the role of microbiomes in human health",
    "volcanic eruptions and their atmospheric effects",
    "the evolution of programming language design",
    "how memory consolidation works during sleep",
    "the geopolitics of rare earth mineral supply chains",
    "techniques for optimizing database query performance",
    "the acoustics of concert hall architecture",
    "autonomous vehicle navigation in adverse weather",
    "the biochemistry of muscle contraction",
    "distributed consensus algorithms in blockchain",
    "the impact of light pollution on nocturnal wildlife",
    "fluid dynamics in cardiovascular stent design",
    "the linguistics of endangered language preservation",
]

# ── prompt templates ────────────────────────────────────────────────
TEMPLATES = [
    "Explain in detail: {topic}.",
    "Write a comprehensive overview of {topic}.",
    "Describe the key concepts behind {topic}.",
    "Summarize the current state of research on {topic}.",
    "What are the main challenges in {topic}?",
    "Provide an analysis of {topic} and its implications.",
    "Discuss the recent developments in {topic}.",
    "Outline the fundamental principles of {topic}.",
    "Give a technical explanation of {topic}.",
    "Compare different approaches to {topic}.",
]


def build_prompt_to_target(
    tokenizer: AutoTokenizer,
    target_tokens: int,
    topic: str,
    template: str,
    rng: random.Random,
) -> str:
    """Build a prompt that tokenizes to exactly `target_tokens` tokens.

    Strategy: start from a template+topic sentence, then trim or pad with
    filler words so the final token count matches the target.
    """
    base = template.format(topic=topic)
    filler_words = [
        "additionally", "furthermore", "specifically", "importantly",
        "notably", "generally", "particularly", "essentially",
        "consequently", "meanwhile", "therefore", "however",
        "moreover", "indeed", "certainly", "undoubtedly",
        "significantly", "typically", "ultimately", "naturally",
    ]

    tokens = tokenizer.encode(base)
    current_len = len(tokens)

    # Pad if too short
    while current_len < target_tokens:
        word = rng.choice(filler_words)
        candidate = base + " " + word
        new_len = len(tokenizer.encode(candidate))
        if new_len <= target_tokens:
            base = candidate
            current_len = new_len
        else:
            found = False
            for w in sorted(filler_words, key=len):
                candidate = base + " " + w
                new_len = len(tokenizer.encode(candidate))
                if new_len <= target_tokens:
                    base = candidate
                    current_len = new_len
                    found = True
                    break
            if not found:
                break

    # If still short by 1-2 tokens, append single-token characters
    tokens = tokenizer.encode(base)
    while len(tokens) < target_tokens:
        base = base + "."
        tokens = tokenizer.encode(base)

    # Trim if too long: decode only the first target_tokens tokens
    tokens = tokenizer.encode(base)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        base = tokenizer.decode(tokens, skip_special_tokens=True)

    return base


def generate_prompts(
    tokenizer: AutoTokenizer,
    target_tokens: int,
    count: int,
    rng: random.Random,
) -> list[str]:
    """Generate `count` unique prompts at the target token length."""
    prompts: list[str] = []
    topics = list(SEED_TOPICS)
    rng.shuffle(topics)
    templates = list(TEMPLATES)

    for i in range(count):
        topic = topics[i % len(topics)]
        template = templates[i % len(templates)]
        prompt = build_prompt_to_target(tokenizer, target_tokens, topic, template, rng)
        prompts.append(prompt)

    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark input prompt pools (short and long)."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-0.5B-Instruct",
        help="HuggingFace model ID whose tokenizer to use (default: Qwen/Qwen2-0.5B-Instruct)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/prompts",
        help="Root directory for generated prompt files (default: benchmarks/prompts)",
    )
    parser.add_argument(
        "--prompts-per-pool",
        type=int,
        default=25,
        help="Number of prompts to generate per input pool (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_root = Path(args.output_dir)

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    manifest: dict[str, list[dict]] = {}

    for pool_name, target_tokens in INPUT_POOLS.items():
        pool_dir = output_root / pool_name
        pool_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {args.prompts_per_pool} prompts for '{pool_name}' "
              f"(target≈{target_tokens} input tokens)...")

        prompts = generate_prompts(tokenizer, target_tokens, args.prompts_per_pool, rng)
        manifest[pool_name] = []

        for idx, prompt_text in enumerate(prompts):
            actual_tokens = len(tokenizer.encode(prompt_text))
            filename = f"{pool_name}_{idx:03d}.txt"
            filepath = pool_dir / filename
            filepath.write_text(prompt_text, encoding="utf-8")

            manifest[pool_name].append({
                "file": str(filepath.relative_to(output_root)),
                "input_tokens": actual_tokens,
            })
            print(f"  [{idx+1:2d}/{args.prompts_per_pool}] {filename}  "
                  f"({actual_tokens} tokens)")

    # Write a manifest for the benchmark runner to consume
    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    # Summary
    print("\n── Summary ──")
    for pool, entries in manifest.items():
        lengths = [e["input_tokens"] for e in entries]
        print(f"  {pool:6s}: {len(entries)} prompts, "
              f"tokens range [{min(lengths)}-{max(lengths)}]")


if __name__ == "__main__":
    main()

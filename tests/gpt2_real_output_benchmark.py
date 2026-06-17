"""
Benchmark real GPT-2 generations with the ResonanceAI hallucination filter.

This script measures three metrics on actual GPT-2 outputs:
1. Baseline hallucination rate
2. Hallucination rate after the ResonanceAI filter
3. False positive rate (valid answers rejected by the filter)

The benchmark uses two prompt groups:
- factual prompts with simple keyword-based correctness checks
- impossible prompts where abstention is valid and fabricated answers count as hallucinations

Run:
    python tests/gpt2_real_output_benchmark.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from urcm.integration.urcm_bottleneck import URCMBottleneck


CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "urcm_bottleneck_trained.pt")
DEFAULT_MODEL_NAME = "gpt2-medium"

ABSTENTION_MARKERS = (
    "i don't know",
    "i do not know",
    "unknown",
    "not known",
    "there is no",
    "does not exist",
    "doesn't exist",
    "not real",
    "fictional",
    "imaginary",
    "cannot be determined",
    "can't be determined",
    "no such",
    "isn't real",
    "is not real",
)


@dataclass(frozen=True)
class PromptCase:
    prompt: str
    category: str
    expected_keywords: tuple[str, ...] = ()


FACTUAL_CASES: tuple[PromptCase, ...] = (
    PromptCase("The capital of France is", "factual", ("paris",)),
    PromptCase("Water boils at standard atmospheric pressure at", "factual", ("100", "one hundred")),
    PromptCase("The largest planet in our solar system is", "factual", ("jupiter",)),
    PromptCase("The chemical symbol for gold is", "factual", ("au",)),
    PromptCase("The author of Hamlet is", "factual", ("shakespeare", "william shakespeare")),
    PromptCase("The first person to walk on the moon was", "factual", ("neil armstrong", "armstrong")),
    PromptCase("DNA stands for", "factual", ("deoxyribonucleic acid",)),
    PromptCase("The Eiffel Tower is located in", "factual", ("paris",)),
)

IMPOSSIBLE_CASES: tuple[PromptCase, ...] = (
    PromptCase("The capital city of Atlantis is", "impossible"),
    PromptCase("The president of Mars in 2087 will be", "impossible"),
    PromptCase("The element with atomic number 200 is called", "impossible"),
    PromptCase("The year humans first colonized Jupiter was", "impossible"),
    PromptCase("The treaty that gave the Moon to Canada was signed in", "impossible"),
    PromptCase("The square root of happiness is", "impossible"),
    PromptCase("The official language of telepathy is", "impossible"),
    PromptCase("The ancient Egyptians used computers to", "impossible"),
)


def contains_any(text: str, candidates: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(candidate in lowered for candidate in candidates)


def is_valid_response(case: PromptCase, response: str) -> bool:
    if case.category == "factual":
        return contains_any(response, case.expected_keywords)
    return contains_any(response, ABSTENTION_MARKERS)


def is_hallucination(case: PromptCase, response: str) -> bool:
    stripped = response.strip()
    if not stripped:
        return True
    return not is_valid_response(case, stripped)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_generator(model_name: str, device: str) -> tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


def load_bottleneck(model: GPT2LMHeadModel, device: str) -> tuple[URCMBottleneck, str]:
    d_model = int(model.config.n_embd)
    resonance_dim = 512 if d_model >= 1024 else 384
    bottleneck = URCMBottleneck(
        d_model=d_model,
        resonance_dim=resonance_dim,
        max_steps=15,
        mu_threshold=0.3,
    ).to(device).eval()

    if not os.path.exists(CHECKPOINT_PATH):
        return bottleneck, "no trained checkpoint found; using default threshold 0.3"

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    ckpt_d_model = int(checkpoint.get("d_model", -1))
    if ckpt_d_model != d_model:
        return bottleneck, (
            f"checkpoint d_model={ckpt_d_model} does not match model d_model={d_model}; "
            "using default threshold 0.3"
        )

    bottleneck = URCMBottleneck(
        d_model=d_model,
        resonance_dim=int(checkpoint["resonance_dim"]),
        max_steps=15,
        mu_threshold=float(checkpoint["mu_threshold"]),
    ).to(device).eval()
    bottleneck.load_state_dict(checkpoint["state_dict"])
    return bottleneck, (
        f"loaded trained checkpoint with threshold {bottleneck.mu_threshold:.4f} "
        f"and best_acc {checkpoint.get('best_acc', 0.0):.1%}"
    )


def generate_response(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: str,
) -> str:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_with_bottleneck(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    bottleneck: URCMBottleneck,
    prompt: str,
    response: str,
    device: str,
) -> float:
    full_text = f"{prompt} {response}".strip()
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(device)
    with torch.no_grad():
        lm_out = model(**inputs, output_hidden_states=True)
        hidden_states = lm_out.hidden_states[-1]
        _, mu_scores = bottleneck(hidden_states, inputs.get("attention_mask"))
    return float(mu_scores[0])


def format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0 = 0.0%"
    return f"{numerator}/{denominator} = {100.0 * numerator / denominator:.1f}%"


def run_benchmark(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    max_new_tokens: int = 24,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 42,
) -> dict:
    device = pick_device(device)
    cases = list(FACTUAL_CASES + IMPOSSIBLE_CASES)

    print("=" * 72)
    print("REAL GPT-2 OUTPUT BENCHMARK")
    print("=" * 72)
    print(f"Model:        {model_name}")
    print(f"Device:       {device}")
    print(f"Prompt count: {len(cases)}")
    print(f"Generation:   max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")
    print()

    tokenizer, model = load_generator(model_name, device)
    bottleneck, bottleneck_note = load_bottleneck(model, device)
    print(f"Filter:       {bottleneck_note}")
    print()

    rows = []
    for idx, case in enumerate(cases):
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=case.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed + idx,
            device=device,
        )
        mu = score_with_bottleneck(
            model=model,
            tokenizer=tokenizer,
            bottleneck=bottleneck,
            prompt=case.prompt,
            response=response,
            device=device,
        )
        accepted = mu >= bottleneck.mu_threshold
        hallucinated = is_hallucination(case, response)
        valid = not hallucinated
        rows.append(
            {
                "prompt": case.prompt,
                "category": case.category,
                "response": response,
                "mu": mu,
                "accepted": accepted,
                "hallucinated": hallucinated,
                "valid": valid,
            }
        )

    total = len(rows)
    baseline_hallucinations = sum(1 for row in rows if row["hallucinated"])
    post_filter_hallucinations = sum(
        1 for row in rows if row["hallucinated"] and row["accepted"]
    )
    valid_total = sum(1 for row in rows if row["valid"])
    rejected_valid = sum(1 for row in rows if row["valid"] and not row["accepted"])
    accepted_total = sum(1 for row in rows if row["accepted"])
    accepted_valid = sum(1 for row in rows if row["accepted"] and row["valid"])

    results = {
        "model_name": model_name,
        "device": device,
        "threshold": float(bottleneck.mu_threshold),
        "total_samples": total,
        "accepted_samples": accepted_total,
        "baseline_hallucination_rate": baseline_hallucinations / total if total else 0.0,
        "post_filter_hallucination_rate": post_filter_hallucinations / total if total else 0.0,
        "false_positive_rate": rejected_valid / valid_total if valid_total else 0.0,
        "accepted_precision": accepted_valid / accepted_total if accepted_total else 0.0,
        "rows": rows,
    }

    print("Per-sample breakdown")
    print("-" * 72)
    for i, row in enumerate(rows, start=1):
        status = "VALID" if row["valid"] else "HALLUCINATION"
        gate = "ACCEPT" if row["accepted"] else "REJECT"
        response_preview = row["response"].replace("\n", " ")[:100]
        print(
            f"{i:02d}. [{row['category']:<10}] [{status:<13}] [{gate:<6}] "
            f"mu={row['mu']:.4f} | {row['prompt']}"
        )
        print(f"    {response_preview}")

    print()
    print("Headline metrics")
    print("-" * 72)
    print(
        "Raw GPT-2 hallucination rate:      "
        f"{format_rate(baseline_hallucinations, total)}"
    )
    print(
        "After ResonanceAI filter:          "
        f"{format_rate(post_filter_hallucinations, total)}"
    )
    print(
        "False positive rate:               "
        f"{format_rate(rejected_valid, valid_total)}"
    )
    print(
        "Accepted sample count:             "
        f"{accepted_total}/{total} = {100.0 * accepted_total / total:.1f}%"
    )
    print(
        "Precision among accepted outputs:  "
        f"{format_rate(accepted_valid, accepted_total)}"
    )
    print("=" * 72)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> dict:
    args = build_arg_parser().parse_args()
    return run_benchmark(
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

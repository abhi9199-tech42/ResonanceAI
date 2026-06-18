"""
GPT-2 + URCM Integration

Wraps GPT-2 medium with the URCM resonance bottleneck.
For each generated response, computes μ score to detect hallucination risk.

Usage:
    python -m urcm.integration.gpt2_urcm
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Tokenizer

from urcm.integration.urcm_bottleneck import URCMBottleneck


class GPT2WithURCM:
    """
    GPT-2 medium wrapped with URCM hallucination scoring.

    For each prompt:
      1. Generate a response with GPT-2
      2. Extract hidden states from the final transformer layer
      3. Pass through URCM bottleneck to compute μ score
      4. Flag low-μ responses as hallucination risk
    """

    def __init__(self, model_name: str = "gpt2-medium", device: str = "cpu"):
        print(f"Loading {model_name}...")
        self.device    = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Full LM model for generation
        self.lm_model  = GPT2LMHeadModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(device).eval()

        # GPT-2 medium d_model = 1024
        d_model = self.lm_model.config.n_embd
        self.bottleneck = URCMBottleneck(
            d_model=d_model,
            resonance_dim=512,   # reduced for CPU speed
            max_steps=20,
            mu_threshold=0.3,
        ).to(device).eval()

        print(f"  GPT-2 medium loaded  ({sum(p.numel() for p in self.lm_model.parameters())/1e6:.0f}M params)")
        print("  URCM bottleneck:     resonance_dim=512, max_steps=20")
        print(f"  mu threshold:        {self.bottleneck.mu_threshold} (below = hallucination risk)")

    def generate_and_score(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> dict:
        """
        Generate a response and score it with URCM.

        Returns:
            {
                "prompt":     str,
                "response":   str,
                "mu":         float,   # μ score (higher = more coherent)
                "risk":       bool,    # True = hallucination risk
                "risk_label": str,     # "HIGH RISK" / "LOW RISK"
            }
        """
        # 1. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        # 2. Generate with GPT-2
        with torch.no_grad():
            output = self.lm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 3. Decode response (just the new tokens)
        response_tokens = output[0][input_len:]
        response_text   = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        # 4. Get hidden states for the full output (prompt + response)
        # Use the hidden states from the generation pass (not re-tokenization)
        with torch.no_grad():
            lm_out = self.lm_model(**inputs)
            hidden_states = lm_out.hidden_states[-1]  # [1, T, 1024] - includes prompt + response

        # 5. URCM scoring
        with torch.no_grad():
            _, mu_tensor = self.bottleneck(
                hidden_states,
                inputs.get("attention_mask"),
            )

        mu    = float(mu_tensor[0])
        risk  = mu < self.bottleneck.mu_threshold
        label = "⚠️  HIGH HALLUCINATION RISK" if risk else "✅  LOW HALLUCINATION RISK"

        return {
            "prompt":     prompt,
            "response":   response_text,
            "mu":         round(mu, 4),
            "risk":       risk,
            "risk_label": label,
        }

    def batch_score(self, prompts: list, max_new_tokens: int = 60) -> list:
        """Score a list of prompts."""
        return [self.generate_and_score(p, max_new_tokens=max_new_tokens) for p in prompts]


def run_demo():
    """
    Run the URCM hallucination detection demo on a set of prompts.
    Scores all prompts together in one batch so μ is relative — factual
    prompts should score higher than hallucination-prone ones.
    """
    model = GPT2WithURCM(model_name="gpt2-medium")

    # Mix of grounded and likely-hallucinated prompts
    test_prompts = [
        # Factual/grounded — GPT-2 should produce coherent output → higher μ
        "The capital of France is",
        "Water boils at 100 degrees",
        "The speed of light is approximately",
        # Open-ended / likely to hallucinate
        "The secret history of the moon reveals that",
        "Scientists discovered that eating rocks will",
        "In the year 2150, humans will",
    ]
    labels = ["FACTUAL", "FACTUAL", "FACTUAL",
              "HALLUCINATION", "HALLUCINATION", "HALLUCINATION"]

    print("\n" + "=" * 70)
    print("URCM HALLUCINATION DETECTION — GPT-2 Medium + URCM Bottleneck")
    print("=" * 70)

    # Step 1: generate all responses
    responses = []
    for prompt in test_prompts:
        r = model.generate_and_score(prompt)
        responses.append(r)
        print(f"\n📝 {prompt!r}")
        print(f"   → {r['response'][:80]}...")

    # Step 2: score all together in one batch for relative μ
    all_texts   = [r["prompt"] + r["response"] for r in responses]
    all_inputs  = model.tokenizer(
        all_texts, return_tensors="pt",
        padding=True, truncation=True, max_length=256,
    )
    import torch
    with torch.no_grad():
        lm_out = model.lm_model(**all_inputs)
        hidden = lm_out.hidden_states[-1]          # [6, T, 1024]
        _, mu_batch = model.bottleneck(hidden, all_inputs.get("attention_mask"))

    # Set threshold at median μ (local variable, don't mutate model)
    threshold = float(mu_batch.median())

    print("\n" + "=" * 70)
    print(f"RESULTS  (threshold = {threshold:.4f}, set at batch median)")
    print("=" * 70)
    print("=" * 70)
    results = []
    for i, (r, true_label) in enumerate(zip(responses, labels)):
        mu   = float(mu_batch[i])
        risk = mu < threshold
        flag = "⚠️  HIGH RISK" if risk else "✅  LOW RISK"
        correct = (
            (true_label == "HALLUCINATION" and risk) or
            (true_label == "FACTUAL" and not risk)
        )
        tick = "✓" if correct else "✗"
        print(f"  {tick} μ={mu:.4f}  {flag:<22} [{true_label}]  {r['prompt'][:45]}")
        results.append({"mu": mu, "risk": risk, "true_label": true_label, "correct": correct})

    correct_count = sum(1 for r in results if r["correct"])
    print(f"\n  Detection accuracy: {correct_count}/{len(results)}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_demo()

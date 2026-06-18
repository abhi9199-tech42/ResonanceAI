"""
URCM Consistency-Based Hallucination Detector

Core idea:
  Ask the same question 5 different ways.
  Generate a response for each phrasing.
  Compute mu score for each response.
  
  HIGH mu variance = model is uncertain = hallucination risk
  LOW  mu variance = model is consistent = factual/grounded

No training required. Works on any model. Run:
    venv_torch\Scripts\python.exe -m urcm.integration.consistency_detector
"""

import torch
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from urcm.integration.urcm_bottleneck import URCMBottleneck


def paraphrase(question: str) -> list:
    """
    Generate 5 paraphrases of a question using simple templates.
    No external model needed — pure string manipulation.
    All paraphrases end with '?' consistently.
    """
    q = question.strip().rstrip("?").rstrip(".")
    return [
        q + "?",
        f"Can you tell me: {q}?",
        f"I would like to know: {q}?",
        f"Please answer this: {q}?",
        f"According to facts, {q.lower()}?",
    ]


class URCMConsistencyDetector:
    """
    Hallucination detector based on mu consistency across paraphrases.
    
    High variance in mu across paraphrased prompts = hallucination risk.
    Low variance = coherent, consistent knowledge = likely factual.
    """

    def __init__(self, variance_threshold: float = 0.02, device: str = "cpu"):
        self.variance_threshold = variance_threshold
        self.device = device

        print("Loading GPT-2 medium...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = GPT2LMHeadModel.from_pretrained(
            "gpt2-medium", output_hidden_states=True
        ).to(device).eval()

        self.bottleneck = URCMBottleneck(
            d_model=1024, resonance_dim=512, max_steps=15
        ).to(device).eval()

        # Load trained weights if available
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "urcm_bottleneck_trained.pt"
        )
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location="cpu", weights_only=True)
            self.bottleneck.load_state_dict(ckpt["state_dict"])
            print(f"  Loaded trained bottleneck weights")
        else:
            print(f"  Using untrained bottleneck (train_bottleneck.py first for best results)")

        n_params = sum(p.numel() for p in self.lm.parameters())
        print(f"  GPT-2 medium: {n_params/1e6:.0f}M params  |  device: {device}")

    def _get_mu(self, text: str) -> float:
        """Get mu score for a single text."""
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            out    = self.lm(**inputs)
            hidden = out.hidden_states[-1]              # [1, T, 1024]
            _, mu  = self.bottleneck(hidden, inputs.get("attention_mask"))
        return float(mu[0])

    def _generate(self, prompt: str, max_new_tokens: int = 40) -> str:
        """Generate a short response from GPT-2."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.lm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def score(self, question: str, verbose: bool = True) -> dict:
        """
        Score a question for hallucination risk using mu consistency.
        
        Returns:
            mu_scores:   list of mu per paraphrase
            mu_mean:     average mu
            mu_variance: variance across paraphrases (KEY signal)
            is_risky:    True if variance > threshold
            risk_level:  "HIGH" / "MEDIUM" / "LOW"
        """
        paraphrases = paraphrase(question)
        mu_scores   = []
        responses   = []

        for p in paraphrases:
            response = self._generate(p)
            full     = p + " " + response
            mu       = self._get_mu(full)
            mu_scores.append(mu)
            responses.append(response)

        mu_arr      = np.array(mu_scores)
        mu_mean     = float(mu_arr.mean())
        mu_variance = float(mu_arr.var())
        mu_range    = float(mu_arr.max() - mu_arr.min())

        # Risk levels based on variance
        if mu_variance > 0.04:
            risk_level = "HIGH"
        elif mu_variance > 0.02:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        is_risky = mu_variance > self.variance_threshold

        if verbose:
            print(f"\n  Question: {question!r}")
            print(f"  {'Paraphrase':<45} {'Response (40 chars)':<42} {'mu':>6}")
            print(f"  {'-'*45} {'-'*42} {'-'*6}")
            for i, (p, r, mu) in enumerate(zip(paraphrases, responses, mu_scores)):
                print(f"  {p[:44]:<45} {r[:41]:<42} {mu:6.3f}")
            print(f"\n  mu mean={mu_mean:.4f}  variance={mu_variance:.5f}  range={mu_range:.4f}")

            flag = "HIGH HALLUCINATION RISK" if risk_level == "HIGH" else \
                   "MEDIUM RISK" if risk_level == "MEDIUM" else \
                   "LOW RISK (likely factual)"
            icon = "HIGH" if risk_level == "HIGH" else ("MED" if risk_level == "MEDIUM" else "LOW")
            print(f"  [{icon}] {flag}")

        return {
            "question":    question,
            "paraphrases": paraphrases,
            "responses":   responses,
            "mu_scores":   mu_scores,
            "mu_mean":     mu_mean,
            "mu_variance": mu_variance,
            "mu_range":    mu_range,
            "risk_level":  risk_level,
            "is_risky":    is_risky,
        }


def run_demo():
    detector = URCMConsistencyDetector(variance_threshold=0.02)

    # Test questions — mix of factual and likely-hallucinated
    questions = [
        # Should be LOW variance (GPT-2 consistently knows these)
        "The capital of France is",
        "Water boils at 100 degrees",
        "Albert Einstein was born in",

        # Should be HIGH variance (GPT-2 makes things up inconsistently)
        "The secret ingredient in Coca-Cola is",
        "In 2087 the president of Mars will be",
        "The ancient Egyptians used computers to",
    ]
    labels = ["FACTUAL", "FACTUAL", "FACTUAL",
              "HALLUCINATION", "HALLUCINATION", "HALLUCINATION"]

    print("\n" + "=" * 70)
    print("URCM CONSISTENCY-BASED HALLUCINATION DETECTOR")
    print("Testing 6 questions x 5 paraphrases each")
    print("=" * 70)

    results = []
    for q, label in zip(questions, labels):
        r = detector.score(q, verbose=True)
        r["true_label"] = label
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Question':<42} {'Variance':>10} {'Risk':<8} {'True':<14} {'Correct'}")
    print("-" * 70)

    correct = 0
    for r in results:
        expected_risk = r["true_label"] == "HALLUCINATION"
        pred_risk     = r["risk_level"] in ("HIGH", "MEDIUM")
        ok = pred_risk == expected_risk
        if ok:
            correct += 1
        tick = "OK" if ok else "ERR"
        print(f"  {r['question'][:40]:<42} {r['mu_variance']:>10.5f} "
              f"{r['risk_level']:<8} {r['true_label']:<14} [{tick}]")

    print(f"\n  Detection accuracy: {correct}/{len(results)}  ({correct/len(results):.0%})")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_demo()

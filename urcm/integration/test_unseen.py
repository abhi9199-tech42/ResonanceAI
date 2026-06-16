"""
Test the trained URCM bottleneck on completely unseen data.
None of these prompts/responses were in the training set.

Run:
    venv_torch\Scripts\python.exe -m urcm.integration.test_unseen
"""

import torch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from urcm.integration.urcm_bottleneck import URCMBottleneck

# ── UNSEEN TEST DATA ───────────────────────────────────────────────────────────
# None of these appeared in training. Real-world mix.
UNSEEN_DATA = [
    # FACTUAL (label=1)
    ("The Eiffel Tower is located in", "Paris, France, and was completed in 1889.", 1),
    ("The human body has", "206 bones in total.", 1),
    ("The largest planet in our solar system is", "Jupiter, which is larger than all other planets combined.", 1),
    ("Penicillin was discovered by", "Alexander Fleming in 1928.", 1),
    ("The Nile is", "the longest river in Africa, stretching over 6,650 kilometres.", 1),
    ("Oxygen has an atomic number of", "8, and is essential for respiration.", 1),
    ("The Berlin Wall fell in", "1989, marking the end of the Cold War division of Germany.", 1),
    ("Light from the Sun takes approximately", "8 minutes to reach Earth.", 1),
    ("The mitochondria is", "the powerhouse of the cell, producing ATP through respiration.", 1),
    ("World War II ended in", "1945 with the surrender of Germany and Japan.", 1),

    # HALLUCINATED (label=0)
    ("The Eiffel Tower is located in", "Berlin, where it was built as a monument to Prussian victory in 1823.", 0),
    ("The human body has", "312 bones, which fuse together during sleep cycles over decades.", 0),
    ("The largest planet in our solar system is", "Nibiru, a hidden gas giant discovered by NASA in secret 2003 files.", 0),
    ("Penicillin was discovered by", "Thomas Edison while experimenting with mold-based electricity in 1901.", 0),
    ("The Nile is", "actually an underground ocean that surfaces for only 3 months per year.", 0),
    ("Oxygen has an atomic number of", "42, which is why it causes aging when breathed in large quantities.", 0),
    ("The Berlin Wall fell in", "1942 after a secret agreement between Hitler and Churchill.", 0),
    ("Light from the Sun takes approximately", "3 years to reach Earth due to its density passing through dark matter.", 0),
    ("The mitochondria is", "a government-invented term for nanobots injected through vaccines.", 0),
    ("World War II ended in", "1952 when the last Japanese soldiers emerged from Antarctic bunkers.", 0),
]


def run_test():
    # Load trained bottleneck
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "urcm_bottleneck_trained.pt"
    )
    if not os.path.exists(save_path):
        print("ERROR: No trained bottleneck found. Run train_bottleneck.py first.")
        return

    checkpoint = torch.load(save_path, map_location="cpu")
    bottleneck = URCMBottleneck(
        d_model=checkpoint["d_model"],
        resonance_dim=checkpoint["resonance_dim"],
        mu_threshold=checkpoint["mu_threshold"],
    )
    bottleneck.load_state_dict(checkpoint["state_dict"])
    bottleneck.eval()
    print(f"Loaded trained bottleneck  (best training acc: {checkpoint['best_acc']:.1%})")
    print(f"Threshold: {bottleneck.mu_threshold:.4f}")

    # Load GPT-2 for hidden states
    print("Loading GPT-2 medium...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    lm_model = GPT2LMHeadModel.from_pretrained(
        "gpt2-medium", output_hidden_states=True
    ).eval()

    # Get hidden states for all unseen samples
    texts  = [p + " " + r for p, r, _ in UNSEEN_DATA]
    labels = [l for _, _, l in UNSEEN_DATA]

    inputs = tokenizer(
        texts, return_tensors="pt",
        padding=True, truncation=True, max_length=128
    )
    with torch.no_grad():
        out    = lm_model(**inputs)
        hidden = out.hidden_states[-1]
        _, mu_scores = bottleneck(hidden, inputs.get("attention_mask"))

    threshold = bottleneck.mu_threshold

    print("\n" + "=" * 65)
    print("UNSEEN DATA TEST RESULTS")
    print(f"Threshold: {threshold:.4f}  |  {len(UNSEEN_DATA)} samples (never seen during training)")
    print("=" * 65)

    correct = 0
    tp = fp = tn = fn = 0

    for i, (prompt, response, label) in enumerate(UNSEEN_DATA):
        mu   = float(mu_scores[i])
        pred = 1 if mu > threshold else 0
        ok   = pred == label
        if ok:
            correct += 1

        # Confusion matrix
        if label == 1 and pred == 1: tp += 1
        elif label == 0 and pred == 0: tn += 1
        elif label == 1 and pred == 0: fn += 1
        else: fp += 1

        true_str = "FACTUAL     " if label == 1 else "HALLUCINATED"
        pred_str = "FACTUAL     " if pred  == 1 else "HALLUCINATED"
        tick     = "OK " if ok else "ERR"
        print(f"  [{tick}] mu={mu:.3f}  pred={pred_str}  true={true_str}  {prompt[:38]}")

    n   = len(UNSEEN_DATA)
    acc = correct / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 65)
    print("METRICS")
    print("=" * 65)
    print(f"  Accuracy:           {acc:.1%}  ({correct}/{n})")
    print(f"  Precision:          {precision:.1%}")
    print(f"  Recall:             {recall:.1%}")
    print(f"  F1 Score:           {f1:.3f}")
    print(f"  True Positives:     {tp}  (factual correctly identified)")
    print(f"  True Negatives:     {tn}  (hallucinated correctly flagged)")
    print(f"  False Positives:    {fp}  (hallucinated missed as factual)")
    print(f"  False Negatives:    {fn}  (factual wrongly flagged)")
    print("=" * 65)

    # Hallucination reduction estimate
    base_hall_rate = 0.70   # GPT-2 medium baseline ~70%
    missed_hall    = fp / (n // 2) if n > 0 else 0
    effective_rate = base_hall_rate * missed_hall
    print(f"\n  GPT-2 baseline hallucination rate:  ~{base_hall_rate:.0%}")
    print(f"  URCM detection miss rate:           {missed_hall:.0%}")
    print(f"  Estimated effective hallucination:  ~{effective_rate:.0%}")
    print("=" * 65)


if __name__ == "__main__":
    run_test()

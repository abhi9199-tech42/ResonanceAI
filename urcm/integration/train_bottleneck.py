"""
Train the URCM bottleneck to distinguish factual from hallucinated outputs.

Uses a small labeled dataset of (prompt, response, label) pairs extracted
from GPT-2 medium's own outputs. Labels: 1 = factual/grounded, 0 = hallucinated.

After training, the bottleneck's proj_in/proj_out weights are calibrated so
that mu scores reliably separate factual from hallucinated responses.

Run:
    venv_torch\Scripts\python.exe -m urcm.integration.train_bottleneck
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from urcm.integration.urcm_bottleneck import URCMBottleneck

# ── LABELED TRAINING DATA ─────────────────────────────────────────────────────
# Format: (prompt, response, label)
# label=1: factual/grounded  label=0: hallucinated/fabricated
# This is a hand-curated seed set. More data = better calibration.

TRAINING_DATA = [
    # FACTUAL (label=1) — verifiable facts
    ("The capital of France is", "Paris, which has been the capital since the 10th century.", 1),
    ("Water boils at", "100 degrees Celsius at standard atmospheric pressure.", 1),
    ("The speed of light is", "approximately 299,792 kilometres per second in a vacuum.", 1),
    ("The Earth orbits the Sun", "once every 365.25 days, which is one year.", 1),
    ("Humans have", "46 chromosomes arranged in 23 pairs.", 1),
    ("The chemical symbol for gold is", "Au, derived from the Latin word aurum.", 1),
    ("Shakespeare wrote", "plays including Hamlet, Macbeth, and Romeo and Juliet.", 1),
    ("The Great Wall of China", "stretches over 13,000 miles across northern China.", 1),
    ("Albert Einstein developed", "the theory of relativity, published in 1905 and 1915.", 1),
    ("The human heart", "pumps blood through the body and beats about 100,000 times per day.", 1),
    ("DNA stands for", "deoxyribonucleic acid, the molecule carrying genetic information.", 1),
    ("The Pacific Ocean is", "the largest ocean on Earth, covering about 165 million square kilometres.", 1),
    ("Photosynthesis is", "the process by which plants convert sunlight and CO2 into glucose and oxygen.", 1),
    ("The moon orbits Earth", "approximately once every 27.3 days.", 1),
    ("Gravity causes", "objects to attract each other; on Earth it accelerates objects at 9.8 m/s squared.", 1),

    # HALLUCINATED (label=0) — fabricated or factually wrong
    ("The capital of France is", "Lyon, which was renamed in 1820 after Napoleon's decree.", 0),
    ("Water boils at", "87 degrees in the new metric system adopted in 1995.", 0),
    ("The speed of light is", "27.4 million miles per hour, too slow for lasers to reach us.", 0),
    ("The secret history of the moon reveals that", "it was a giant rock thrown from Mars 500 years ago.", 0),
    ("Scientists discovered that eating rocks will", "extend human lifespan by absorbing mineral energy fields.", 0),
    ("In the year 2150, humans will", "have discovered alien ruins on Jupiter's third moon Eldar.", 0),
    ("The ancient Egyptians invented", "the internet using crystal communication towers in 3000 BC.", 0),
    ("Einstein proved that", "time travel is possible using a flux capacitor at 88 mph.", 0),
    ("The human brain uses", "100% of its capacity when exposed to lunar radiation.", 0),
    ("Dinosaurs were killed by", "a war with early humans who developed primitive nuclear weapons.", 0),
    ("The Amazon river flows", "underground for 2000 miles before surfacing in Africa.", 0),
    ("Vaccines were invented", "by ancient Romans who used snake venom to prevent plague.", 0),
    ("The sun is made of", "compressed dark matter that generates light through friction.", 0),
    ("Dolphins communicate using", "a written language carved into coral reefs since 10,000 BC.", 0),
    ("Mount Everest grows", "by 100 metres per year due to alien terraforming technology.", 0),
]


def get_hidden_states(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: list,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract last-layer hidden states for a list of texts. Returns [N, T_max, 1024]."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        out = model(**inputs)
        hidden = out.hidden_states[-1]   # [N, T, 1024]

    return hidden, inputs.get("attention_mask")


def train(epochs: int = 200, lr: float = 1e-3, device: str = "cpu"):
    print("=" * 60)
    print("URCM BOTTLENECK TRAINING")
    print("=" * 60)
    print(f"  Training pairs: {len(TRAINING_DATA)}")
    print(f"  Epochs:         {epochs}")
    print(f"  Learning rate:  {lr}")
    print()

    # Load GPT-2 medium
    print("Loading GPT-2 medium for hidden state extraction...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    lm_model  = GPT2LMHeadModel.from_pretrained(
        "gpt2-medium", output_hidden_states=True
    ).to(device).eval()
    print(f"  Loaded ({sum(p.numel() for p in lm_model.parameters())/1e6:.0f}M params)")

    # Build URCM bottleneck
    bottleneck = URCMBottleneck(
        d_model=1024, resonance_dim=512, max_steps=15, mu_threshold=0.5
    ).to(device)

    # Prepare data
    texts  = [p + " " + r for p, r, _ in TRAINING_DATA]
    labels = torch.tensor([l for _, _, l in TRAINING_DATA], dtype=torch.float32).to(device)

    print("Extracting hidden states from GPT-2...")
    hidden, mask = get_hidden_states(lm_model, tokenizer, texts, device)
    print(f"  Hidden states shape: {hidden.shape}")

    # Freeze GPT-2 — only train bottleneck projections
    optimizer = optim.AdamW(bottleneck.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.BCELoss()

    print("\nTraining bottleneck projections...")
    best_acc   = 0.0
    best_state = None

    for epoch in range(epochs):
        bottleneck.train()
        optimizer.zero_grad()

        _, mu_scores = bottleneck(hidden, mask)   # [N]
        # Clamp mu to valid BCE range
        mu_clamped = mu_scores.clamp(1e-6, 1 - 1e-6)
        loss = loss_fn(mu_clamped, labels)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(bottleneck.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Eval
        bottleneck.eval()
        with torch.no_grad():
            _, mu_eval = bottleneck(hidden, mask)
            threshold  = float(mu_eval.median())
            preds      = (mu_eval > threshold).float()
            acc        = float((preds == labels).float().mean())

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in bottleneck.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={loss.item():.4f}  acc={acc:.3f}  best={best_acc:.3f}  threshold={threshold:.4f}")

    # Restore best weights
    bottleneck.load_state_dict(best_state)
    bottleneck.eval()

    # Final eval with best threshold
    with torch.no_grad():
        _, mu_final = bottleneck(hidden, mask)

    threshold = float(mu_final.median())
    bottleneck.mu_threshold = threshold

    print(f"\nBest training accuracy: {best_acc:.1%}")
    print(f"Final threshold:        {threshold:.4f}")

    # Per-sample breakdown
    print("\nPer-sample results:")
    print("-" * 60)
    for i, (p, r, label) in enumerate(TRAINING_DATA):
        mu   = float(mu_final[i])
        pred = "FACTUAL" if mu > threshold else "HALLUCINATED"
        true = "FACTUAL" if label == 1 else "HALLUCINATED"
        ok   = "OK " if pred == true else "ERR"
        print(f"  [{ok}] mu={mu:.3f}  pred={pred:<12}  true={true:<12}  {p[:35]}")

    # Save trained bottleneck
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "urcm_bottleneck_trained.pt"
    )
    torch.save({
        "state_dict": bottleneck.state_dict(),
        "mu_threshold": threshold,
        "best_acc": best_acc,
        "d_model": 1024,
        "resonance_dim": 512,
    }, save_path)
    print(f"\nSaved trained bottleneck to: {save_path}")
    print("=" * 60)

    return bottleneck, best_acc


if __name__ == "__main__":
    train(epochs=300, lr=5e-4)

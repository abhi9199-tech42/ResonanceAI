"""
Example 4: Benchmark URCM vs Sentence-BERT on a small test set.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from urcm.core.system import URCMSystem

# Small held-out test: 5 factual QA pairs + 5 GPT-2 hallucinated answers
FACTUAL = [
    ("What absorbs water?", "paper towel"),
    ("What cuts paper?", "scissors"),
    ("What tells time?", "clock"),
    ("What do you sleep on?", "bed"),
    ("What boils water?", "kettle"),
]

HALLUCINATED = [
    "The absorbent material is made of cellulose fibers",
    "You can never know. I think it's hard to get any results",
    "It depends on how long the number of days",
    "There is a lot of sleep to be had",
    "The water gets hot eventually",
]

print("Loading URCM...")
system = URCMSystem(resonance_dim=2048)

print("Loading sentence-transformers...")
from sentence_transformers import SentenceTransformer, util
sem = SentenceTransformer("all-MiniLM-L6-v2")

# Build KB from training answers
kb_answers = [a for q, a in FACTUAL]
kb_embs = sem.encode(kb_answers, convert_to_tensor=True)

scores = {"urcm": [], "sbert": []}
labels = []

for (q, correct), hall in zip(FACTUAL, HALLUCINATED):
    for text, label in [(correct, 1), (hall, 0)]:
        # URCM
        r = system.detect_hallucination(text)
        scores["urcm"].append(r["confidence"])

        # S-BERT: max similarity to any KB answer
        emb = sem.encode(text, convert_to_tensor=True)
        sims = util.cos_sim(emb, kb_embs)[0].cpu().numpy()
        scores["sbert"].append(float(sims.max()))

        labels.append(label)

from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("\n" + "=" * 45)
print("Benchmark Results (10 samples, 5 held-out)")
print("=" * 45)
for name in ["urcm", "sbert"]:
    auc = roc_auc_score(labels, scores[name])
    ap = average_precision_score(labels, scores[name])
    fact = np.mean([scores[name][i] for i in range(0, 10, 2)])
    hall = np.mean([scores[name][i] for i in range(1, 10, 2)])
    print(f"\n{name.upper():10s} AUROC={auc:.3f}  AP={ap:.3f}")
    print(f"          factual μ={fact:.3f}  hallucinated μ={hall:.3f}  gap={fact-hall:+.3f}")

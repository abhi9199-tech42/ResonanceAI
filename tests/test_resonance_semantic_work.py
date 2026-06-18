"""
Diagnostic: does raw resonance do semantic work, or are keyword rules carrying the weight?
Run this before starting any gap fix.
"""
import numpy as np
import pytest
from urcm.core.system import URCMSystem


def raw_resonance_qa(system, question, choices):
    """Pure cosine similarity — no rules, no Hebbian, no qa_w."""
    q_path = system.pipeline.process_text(question)
    q_vec = system.encoder.get_resonance_state(q_path).resonance_vector

    scores = []
    for c in choices:
        c_path = system.pipeline.process_text(c)
        c_vec = system.encoder.get_resonance_state(c_path).resonance_vector
        sim = float(np.dot(q_vec, c_vec) / (
            np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9
        ))
        scores.append(sim)
    return int(np.argmax(scores))


DATASET = [
    {"q": "What do people use to absorb water?",
     "choices": ["spoon", "paper towel", "plate", "pen", "computer"],
     "answer_idx": 1},
    {"q": "Where do you store dishes in a kitchen?",
     "choices": ["cupboard", "trash can", "backpack", "street", "bed"],
     "answer_idx": 0},
    {"q": "What do you use to cut paper?",
     "choices": ["scissors", "spoon", "plate", "rope", "glue"],
     "answer_idx": 0},
]


def test_raw_resonance_diagnostic():
    """
    NOT a pass/fail correctness test — a diagnostic.
    Prints score and tells you which gap to fix first.
    """
    system = URCMSystem(resonance_dim=1024)

    ok = 0
    for d in DATASET:
        pred = raw_resonance_qa(system, d["q"], d["choices"])
        correct = pred == d["answer_idx"]
        ok += int(correct)
        status = "PASS" if correct else "FAIL"
        print(f"  [{status}]  Q: {d['q'][:45]}...")
        print(f"      Predicted: {d['choices'][pred]}, Expected: {d['choices'][d['answer_idx']]}")

    print(f"\n  Raw resonance score (no rules): {ok}/3")

    if ok == 0:
        print("\n  VERDICT: Resonance alone is random — keyword rules carry 100% of QA accuracy.")
        print("  START WITH: Gap 2 (train W_res with Hebbian deposits)")
        print("  Run: python -m urcm.cli warmup --dim 1024")
    elif ok == 1:
        print("\n  VERDICT: Resonance has weak signal — rules are doing most of the work.")
        print("  START WITH: Gap 2 (extend warmup to 200+ QA pairs, then retrain)")
    elif ok == 2:
        print("\n  VERDICT: Resonance has meaningful signal — rules are a useful boost.")
        print("  START WITH: Gap 1 (replace BrocaArea) — quick win, resonance is solid enough")
    else:
        print("\n  VERDICT: Resonance works well on its own.")
        print("  START WITH: Gap 1 (replace BrocaArea) — rules can be stripped safely")

    assert 0 <= ok <= len(DATASET), f"Score {ok} out of range for {len(DATASET)} items"

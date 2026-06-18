
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from urcm.core.reasoning import ReasoningEngine


def verify_grammar():
    print("🧠 LOADING BRAIN FOR GRAMMAR CHECK...")
    try:
        engine = ReasoningEngine()
    except Exception as e:
        print(f"❌ Failed to load brain (might be locked): {e}")
        return

    print(f"✅ Brain Loaded. Vocab: {len(engine.concept_map)}")

    # Test Subject-Verb-Object patterns
    # We look at what follows common subjects

    subjects = ["holmes", "watson", "the", "it", "he", "she", "door"]

    print("\n🔍 CHECKING NEXT-TOKEN PREDICTIONS (GRAMMAR)...")

    for subj in subjects:
        if subj not in engine.concept_map:
            print(f"  ⚠️ '{subj}' not in vocabulary yet.")
            continue

        vec = engine.concept_map[subj]

        # Predict next
        pred_vec = np.tanh(np.dot(vec, engine.hierarchy.layer2.W_res))

        # Find top 3
        dists = []
        for w, v in engine.concept_map.items():
            d = np.linalg.norm(pred_vec - v)
            dists.append((w, d))

        dists.sort(key=lambda x: x[1])
        top_3 = [x[0] for x in dists[:3]]

        print(f"  '{subj}' -> {top_3}")

    # Check "The [Noun]" pattern
    # Does 'the' predict nouns?

    # Check "He [Verb]" pattern
    # Does 'he' predict verbs?

if __name__ == "__main__":
    verify_grammar()

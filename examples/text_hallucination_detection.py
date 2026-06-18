"""
Example 1: Basic hallucination detection on text.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urcm.core.system import URCMSystem

# Load with 62-pair trained weights (phoneme-based, deterministic)
system = URCMSystem(resonance_dim=2048)

# Known correct answers from the training set
known_correct = [
    "paper towel",    # "What absorbs water?"
    "scissors",       # "What cuts paper?"
    "clock",          # "What tells time?"
    "kettle",         # "What boils water?"
    "Paris",          # "What is the capital of France?"
]

# GPT-2 generated hallucinated answers (plausible but wrong)
hallucinated = [
    "The absorbent material is made of cellulose fibers",
    "You can never know. I think it's hard to get any results",
    "It depends on how long the number of days",
    "The water gets hot eventually",
    "Lyon is the capital of France I think",
]

print("=" * 55)
print("URCM Hallucination Detection")
print("=" * 55)

print("\n--- Known correct answers (should score HIGH) ---")
for text in known_correct:
    result = system.detect_hallucination(text)
    label = "CORRECT" if result["confidence"] > 0.65 else "UNCERTAIN"
    print(f"  '{text:50s}'  conf={result['confidence']:.3f}  {label}")

print("\n--- GPT-2 hallucinated answers (should score LOW) ---")
for text in hallucinated:
    result = system.detect_hallucination(text)
    label = "HALLUCINATION" if result["confidence"] < 0.65 else "UNCERTAIN"
    print(f"  '{text:50s}'  conf={result['confidence']:.3f}  {label}")

print("\n--- Raw output example ---")
result = system.detect_hallucination("paper towel", top_k=3)
for key in ["confidence", "raw_cosine", "nn_label", "resonance_norm", "input_length"]:
    print(f"  {key}: {result[key]}")
print(f"  top_k_labels: {result['top_k_labels']}")

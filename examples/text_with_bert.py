"""
Example 2: Using BERT-converted weights for improved detection.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urcm.core.system import URCMSystem

# Load with BERT-converted weights (requires torch + transformers)
# These are cached after first conversion
print("Loading URCM with BERT weights...")
system = URCMSystem(resonance_dim=2048, load_pretrained="bert-base-uncased")

test_cases = [
    # (label, text)
    ("correct", "paper towel"),
    ("correct", "1945"),           # WW2 end year
    ("correct", "Paris"),
    ("correct", "scissors"),
    ("hallucinated", "The capital of France is London"),
    ("hallucinated", "Water boils at 200 degrees Celsius"),
    ("hallucinated", "I think the answer might be something else"),
    ("nonsense", "asdfghjkl qwertyuiop"),
]

print(f"\n{'Label':20s} {'Text':40s} {'Conf':>6s} {'Raw':>6s}")
print("-" * 75)

for label, text in test_cases:
    r = system.detect_hallucination(text)
    print(f"{label:20s} {text[:38]:40s} {r['confidence']:.3f} {r.get('raw_cosine', 0):.3f}")

print("\nNote: BERT weights give richer resonance dynamics but same")
print("fundamental behavior — scores are driven by phoneme pattern")
print("similarity to hippocampus entries, not semantic understanding.")

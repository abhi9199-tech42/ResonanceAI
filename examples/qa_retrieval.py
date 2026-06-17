"""
Example 3: QA via hippocampus concept retrieval (multiple choice).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urcm.core.system import URCMSystem

system = URCMSystem(resonance_dim=2048)

qa_pairs = [
    ("What absorbs water?", ["spoon", "paper towel", "plate", "pen", "computer"], "paper towel"),
    ("What cuts paper?", ["scissors", "spoon", "plate", "rope", "glue"], "scissors"),
    ("Where do you keep milk cold?", ["oven", "refrigerator", "desk", "closet", "backpack"], "refrigerator"),
    ("What do you sleep on?", ["chair", "bed", "table", "floor", "car"], "bed"),
    ("What boils water?", ["kettle", "cup", "plate", "spoon", "book"], "kettle"),
]

print("=" * 55)
print("URCM Concept Retrieval (Multiple Choice QA)")
print("=" * 55)

correct = 0
for question, choices, expected in qa_pairs:
    result = system.solve_qa_right_brain(question, choices)
    winner = result.get("winner", "N/A")
    is_correct = winner == expected
    if is_correct:
        correct += 1
    mark = "[OK]" if is_correct else "[XX]"
    print(f"\nQ: {question}")
    print(f"   Choices: {choices}")
    print(f"   {mark} Answer: {winner}  (expected: {expected})")
    if "details" in result:
        for d in result["details"]:
            print(f"      {d['choice']:15s} score={d['score']:.3f}")

print(f"\nAccuracy: {correct}/{len(qa_pairs)}")

import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from urcm.core.reasoning import ReasoningEngine


def verify_brain():
    print("🧠 LOADING BRAIN FOR VERIFICATION...")
    try:
        engine = ReasoningEngine()
    except Exception as e:
        print(f"❌ Failed to load brain: {e}")
        return

    # Verify hierarchy structure exists
    if not hasattr(engine, 'hierarchy') or not hasattr(engine.hierarchy, 'layer2'):
        print("❌ Engine hierarchy/layer2 not available")
        return
    if not hasattr(engine.hierarchy.layer2, 'W_res'):
        print("❌ W_res not available in layer2")
        return

    print(f"✅ Brain Loaded. Dimensions: {engine.l2_dim}")
    print(f"📚 Vocabulary Size: {len(engine.concept_map)}")

    # 1. Verify Concepts
    print("\n🔍 CHECKING CONCEPTS...")
    expected_concepts = [
        "sky", "blue", "sun", "bright", "fire", "hot", "water", "wet",
        "cats", "animals", "dogs", "rains", "ground", "night", "dark", "day", "light"
    ]

    found = 0
    for c in expected_concepts:
        if c in engine.concept_map:
            found += 1
            # print(f"  ✅ Found '{c}'")
        else:
            print(f"  ❌ Missing '{c}'")

    print(f"👉 Concept Coverage: {found}/{len(expected_concepts)}")

    # 2. Verify Transitions (Logical Flow)
    print("\n🌊 CHECKING LOGIC FLOW (Associative Recall)...")

    test_cases = [
        ("sky", ["blue", "is"]),
        ("sun", ["bright", "is", "shines", "day"]),
        ("fire", ["hot", "is"]),
        ("water", ["wet", "is"]),
        ("cats", ["animals", "are"]),
        ("dogs", ["animals", "are"]),
        ("night", ["dark", "is"]),
        ("day", ["light", "is"])
    ]

    passed = 0
    for start_word, expected_next in test_cases:
        if start_word not in engine.concept_map:
            print(f"  ⚠️ Skipping '{start_word}' (Unknown concept)")
            continue

        # Get Vector
        vec = engine.concept_map[start_word]

        # Predict Next State
        # next_state = tanh(vec * W_res)
        pred_vec = np.tanh(np.dot(vec, engine.hierarchy.layer2.W_res))

        # Decode
        predicted_word = engine.decode(pred_vec)

        # Check Top-N neighbors to be fair
        # (Since 'is' might dominate, we look for semantic relevance nearby)
        dists = []
        for w, v in engine.concept_map.items():
            d = np.linalg.norm(pred_vec - v)
            dists.append((w, d))

        dists.sort(key=lambda x: x[1])
        top_5 = [x[0] for x in dists[:5]]

        # Check match
        match = False
        for exp in expected_next:
            if exp in top_5:
                match = True
                break

        if match:
            print(f"  ✅ {start_word.upper()} -> {predicted_word} (Top 5: {top_5})")
            passed += 1
        else:
            print(f"  ❌ {start_word.upper()} -> {predicted_word} (Expected: {expected_next}) (Top 5: {top_5})")

    print(f"👉 Logic Accuracy: {passed}/{len(test_cases)}")

    # 3. Interactive Chat (Simulated)
    print("\n💬 SIMULATED THOUGHT STREAMS")

    chat_words = ["fire", "sky", "cats"]

    for user_input in chat_words:
        if user_input not in engine.concept_map:
            print(f"   ⚠️ I don't know '{user_input}'.")
            continue

        # Generate Stream of Thought
        current = engine.concept_map[user_input]
        path = [user_input]

        print(f"   Thinking: {user_input}", end="", flush=True)

        for _ in range(5):
            current = np.tanh(np.dot(current, engine.hierarchy.layer2.W_res))
            word = engine.decode(current)
            path.append(word)
            print(f" -> {word}", end="", flush=True)
            # time.sleep(0.1) # Removed sleep for speed

        print("\n")

if __name__ == "__main__":
    verify_brain()

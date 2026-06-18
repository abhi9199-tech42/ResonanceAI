
import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from urcm.core.reasoning import ReasoningEngine

def monitor():
    print("🕵️ MONITORING SHERLOCK'S BRAIN...")
    
    # Check specific Sherlock-related concepts
    watch_words = ["sherlock", "holmes", "watson", "doctor", "crime", "detective", "mystery", "police"]
    
    try:
        engine = ReasoningEngine()
    except Exception:
        print("Brain is locked/writing. Retrying...")
        return

    print(f"🧠 Vocabulary: {len(engine.concept_map)} words")
    print("-" * 40)
    
    for word in watch_words:
        if word not in engine.concept_map:
            print(f"❌ '{word}' not yet learned.")
            continue
            
        vec = engine.concept_map[word]
        
        # Predict what follows this word (Forward Association)
        # Next State = Tanh(Current * W)
        pred_vec = np.tanh(np.dot(vec, engine.hierarchy.layer2.W_res))
        
        # Find nearest neighbors to prediction
        dists = []
        for w, v in engine.concept_map.items():
            if w == word: continue # Skip self
            d = np.linalg.norm(pred_vec - v)
            dists.append((w, d))
            
        dists.sort(key=lambda x: x[1])
        top_3 = [x[0] for x in dists[:3]]
        
        print(f"👉 '{word}' -> predicts -> {top_3}")
        
    print("-" * 40)
    print("Checking specific facts:")
    
    # Does 'Holmes' predict 'Sherlock'? (Reverse association might be weak, usually Sherlock -> Holmes)
    if "sherlock" in engine.concept_map and "holmes" in engine.concept_map:
        v_s = engine.concept_map["sherlock"]
        v_h = engine.concept_map["holmes"]
        pred = np.tanh(np.dot(v_s, engine.hierarchy.layer2.W_res))
        dist = np.linalg.norm(pred - v_h)
        print(f"  Link: Sherlock -> Holmes (Dist: {dist:.4f}) {'✅ Strong' if dist < 1.0 else '⚠️ Weak'}")

if __name__ == "__main__":
    monitor()

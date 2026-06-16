import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from urcm.core.reasoning import ReasoningEngine
from urcm.core.sanskrit_bridge import SanskritBridge

def train_from_file(file_path: str):
    """
    Ingests a text file and performs massive training on the URCM Brain.
    Uses Hebbian learning to associate sequential concepts.
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"🚀 Initializing Massive Training Module...")
    print(f"📂 Source: {file_path}")
    
    # Initialize Engine (Auto-creates 1024-dim brain if missing)
    engine = ReasoningEngine()
    
    # ⚠️ CRITICAL FIX: Zero out W_res for clean slate learning
    # Random initialization (ESN style) is good for chaos, but bad for rote memorization.
    # For massive training, we want to learn specific transitions, not filter noise.
    print("🧹 Wiping W_res to Zero for Tabula Rasa learning...")
    engine.hierarchy.layer2.W_res = np.zeros((engine.l2_dim, engine.l2_dim))
    
    bridge = SanskritBridge()
    
    print(f"🧠 Brain State: {engine.l2_dim} dimensions (Capacity: ~1M Concepts)")
    
    # Read File
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # Preprocessing
    # Split into sentences (simple period split)
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    print(f"📚 Corpus Size: {len(sentences)} sentences.")
    
    # Auto-Epoch for small datasets
    epochs = 10 # Increased for better stability
    lr = 0.1
    if len(sentences) < 200:
        epochs = 30
        lr = 0.5
        print(f"⚠️ Small corpus detected. Boosting to {epochs} epochs and LR={lr}.")
    else:
        print(f"🔄 Standard Training: {epochs} epochs.")
    
    start_time = time.time()
    last_save_time = start_time
    total_updates = 0
    new_concepts = 0
    
    # Stop Words for Skip-Gram
    STOP_WORDS = set(['is', 'a', 'the', 'an', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'has', 'are'])

    for epoch in range(epochs):
        epoch_start_time = time.time() # Reset time per epoch for correct rate
        
        # ---------------------------------------------------------
        # Phase Control
        # Phase 1: Embedding (Geometry) - First 50% of epochs
        # Phase 2: Transition (Dynamics) - Last 50% of epochs
        # ---------------------------------------------------------
        phase = "EMBEDDING" if epoch < (epochs // 2) else "TRANSITION"
        if epochs > 1:
            print(f"🔄 Epoch {epoch+1}/{epochs} [{phase} PHASE]")
            
        for i, sent in enumerate(sentences):
            # Tokenize (Simple space split, remove punctuation)
            clean_sent = "".join([c if c.isalnum() or c.isspace() else "" for c in sent])
            words = [w.lower() for w in clean_sent.split() if w]
            
            if len(words) < 2:
                continue
                
            # ---------------------------------------------------------
            # 1. Ensure Concepts Exist (Hebbian Growth)
            # ---------------------------------------------------------
            for w in words:
                if w not in engine.concept_map:
                    # Use distinct random initialization for each word to prevent Concept Collapse
                    # Seeding with word hash ensures determinism
                    seed = sum(ord(c) for c in w) * 31337
                    np.random.seed(seed % (2**32 - 1))
                    vec = np.random.randn(engine.l2_dim)
                    vec = vec / np.linalg.norm(vec)
                    engine.concept_map[w] = vec
                    new_concepts += 1

            # ---------------------------------------------------------
            # 2. Skip-Gram Training (Window Context)
            # ---------------------------------------------------------
            # Look ahead window size 4 to skip "is a" and hit "detective"
            WINDOW_SIZE = 4
            
            for j in range(len(words)):
                w1 = words[j]
                
                # Iterate forward in window
                for k in range(1, WINDOW_SIZE + 1):
                    if j + k >= len(words):
                        break
                        
                    w2 = words[j+k]
                    
                    # Distance decay factor (closer words = stronger link)
                    dist_factor = 1.0 / k 
                    
                    # A. Transition Learning (Sequence) - ONLY IN PHASE 2
                    # STRICT RULE: Never train transition TO a stop word.
                    # This prevents "Doctor" -> "a" -> "hospital" chains.
                    # We want "Doctor" -> "Hospital".
                    is_immediate = (k == 1)
                    is_w2_content = (w2 not in STOP_WORDS)
                    
                    if phase == "TRANSITION":
                        # Only learn transition if Target is CONTENT
                        if is_w2_content:
                            v1 = engine.concept_map[w1]
                            v2 = engine.concept_map[w2]
                            
                            # Prevent Self-Loops: If vectors are too similar, don't learn transition
                            # This fixes "Sherlock" -> "Sherlock"
                            sim = np.dot(v1, v2)
                            if sim < 0.99: 
                                # Boost LR significantly for clear content jumps
                                current_lr = lr * dist_factor * 5.0 
                                engine.learn_transition(v1, v2, learning_rate=current_lr)
                                total_updates += 1

                    # B. Vector Attraction (Clustering) - ONLY IN PHASE 1
                    if phase == "EMBEDDING":
                        if is_w2_content and w1 not in STOP_WORDS:
                            v1 = engine.concept_map[w1]
                            v2 = engine.concept_map[w2]
                            
                            # Prevent Collapse: Only attract if not already identical
                            if np.dot(v1, v2) < 0.95:
                                attraction_rate = 0.05 * dist_factor
                                v1_new = v1 + attraction_rate * (v2 - v1)
                                v2_new = v2 + attraction_rate * (v1 - v2)
                                
                                engine.concept_map[w1] = v1_new / np.linalg.norm(v1_new)
                                engine.concept_map[w2] = v2_new / np.linalg.norm(v2_new)
            
            # ---------------------------------------------------------
            # 4. Explicit Analogy Training - ONLY IN PHASE 1
            # ---------------------------------------------------------
            if phase == "EMBEDDING":
                # Check for pattern: "A is to B as C is to D"
                if len(words) == 9 and words[1]=="is" and words[2]=="to" and words[4]=="as" and words[6]=="is" and words[7]=="to":
                    # A=0, B=3, C=5, D=8
                    wa, wb, wc, wd = words[0], words[3], words[5], words[8]
                    if all(w in engine.concept_map for w in [wa, wb, wc, wd]):
                        va = engine.concept_map[wa]
                        vb = engine.concept_map[wb]
                        vc = engine.concept_map[wc]
                        vd = engine.concept_map[wd]
                        
                        # Target: D = C + (B - A)
                        # Error: (C + B - A) - D
                        target_d = vc + vb - va
                        
                        # Nudge D towards target
                        analogy_lr = 0.2
                        vd_new = vd + analogy_lr * (target_d - vd)
                        vd_new = vd_new / np.linalg.norm(vd_new)
                        engine.concept_map[wd] = vd_new
                        
                        if epoch == 0:
                            print(f"  ✨ Analogy Taught: {wa}:{wb} :: {wc}:{wd}")

            # ---------------------------------------------------------
            # 5. Explicit Synonym/Antonym Training - ONLY IN PHASE 1
            # ---------------------------------------------------------
            if phase == "EMBEDDING":
                # "X is similar to Y"
                if len(words) == 5 and words[1]=="is" and words[2]=="similar" and words[3]=="to":
                     w1, w2 = words[0], words[4]
                     if w1 in engine.concept_map and w2 in engine.concept_map:
                         v1 = engine.concept_map[w1]
                         v2 = engine.concept_map[w2]
                         # Strong attraction
                         syn_lr = 0.2
                         v1_new = v1 + syn_lr * (v2 - v1)
                         v2_new = v2 + syn_lr * (v1 - v2)
                         engine.concept_map[w1] = v1_new / np.linalg.norm(v1_new)
                         engine.concept_map[w2] = v2_new / np.linalg.norm(v2_new)
                         if epoch == 0:
                             print(f"  🔗 Synonym Taught: {w1} ≈ {w2}")

                # "X is opposite of Y" or "X is opposite to Y" (handle 'of')
                if len(words) >= 5 and words[1]=="is" and words[2]=="opposite":
                     w1, w2 = words[0], words[-1] # Simple parse
                     if w1 in engine.concept_map and w2 in engine.concept_map:
                         v1 = engine.concept_map[w1]
                         v2 = engine.concept_map[w2]
                         # Repulsion
                         opp_lr = 0.2
                         # Move apart: v1 -= lr * v2
                         v1_new = v1 - opp_lr * v2
                         v2_new = v2 - opp_lr * v1
                         engine.concept_map[w1] = v1_new / np.linalg.norm(v1_new)
                         engine.concept_map[w2] = v2_new / np.linalg.norm(v2_new)
                         if epoch == 0:
                             print(f"  ↔️ Antonym Taught: {w1} ≠ {w2}")

            if (i + 1) % 50 == 0:
                elapsed = time.time() - epoch_start_time
                rate = (i + 1) / elapsed
                # Use carriage return for cleaner output
                print(f"\r⚡ Processed {i+1}/{len(sentences)} sentences ({rate:.1f} sent/s) | New Concepts: {new_concepts}", end="")
                sys.stdout.flush()
                
            # Periodic Save (Time-based Checkpointing to prevent IO jam)
            # Save every 5 minutes (300 seconds) instead of every N lines
            current_time = time.time()
            if current_time - last_save_time > 300:
                print("\n💾 Autosaving Checkpoint...")
                engine.save_brain()
                last_save_time = current_time
                print("✅ Checkpoint saved. Resuming...")

    # Final Save
    print("\n💾 Finalizing Brain State...")
    engine.save_brain()
    duration = time.time() - start_time
    print(f"\n✅ TRAINING COMPLETE.")
    print(f"⏱️  Duration: {duration:.2f}s")
    print(f"📈 Total Updates: {total_updates}")
    print(f"🧠 Vocabulary Size: {len(engine.concept_map)}")
    print(f"💾 Brain saved to {engine.brain_data_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python urcm/tools/train_massive.py <path_to_text_file>")
        print("Example: python urcm/tools/train_massive.py corpus.txt")
    else:
        train_from_file(sys.argv[1])

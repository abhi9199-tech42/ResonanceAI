"""Full system health check — parameter count, stability, QA accuracy."""
import numpy as np
import pickle
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.system import URCMSystem

# ── 1. Parameter count ───────────────────────────────────────────────────────
pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
rpenc    = ResonancePathEncoder(input_dim=24, resonance_dim=1024)

total = rpenc.W_in.size + rpenc.W_res.size + rpenc.W_out.size + rpenc.bias.size
print("=" * 50)
print("PARAMETER COUNT")
print("=" * 50)
print(f"  W_in   {rpenc.W_in.shape}   = {rpenc.W_in.size:>10,}")
print(f"  W_res  {rpenc.W_res.shape}  = {rpenc.W_res.size:>10,}")
print(f"  W_out  {rpenc.W_out.shape}   = {rpenc.W_out.size:>10,}")
print(f"  bias   {rpenc.bias.shape}     = {rpenc.bias.size:>10,}")
print(f"  TOTAL                    = {total:>10,}")

# ── 2. Weight file info ──────────────────────────────────────────────────────
import os
wp = "urcm_weights.pkl"
size_mb = os.path.getsize(wp) / 1e6
with open(wp, "rb") as f:
    w = pickle.load(f)
hippo = len(w.get("hippocampus", []))

print()
print("=" * 50)
print("WEIGHT FILE")
print("=" * 50)
print(f"  File size:          {size_mb:.1f} MB")
print(f"  Hippocampus entries: {hippo}")
print(f"  W_res shape:        {w['W_res'].shape}")

# ── 3. Stability check ───────────────────────────────────────────────────────
eigs = np.abs(np.linalg.eigvals(w["W_res"]))
sr   = float(np.max(eigs))
print()
print("=" * 50)
print("STABILITY")
print("=" * 50)
print(f"  Spectral radius:    {sr:.4f}  ({'OK - stable' if sr < 1.5 else 'WARNING - may be unstable'})")

# ── 4. Raw resonance QA (no rules) ──────────────────────────────────────────
print()
print("=" * 50)
print("RAW RESONANCE QA (no keyword rules)")
print("=" * 50)

eval_set = [
    ("What do people use to absorb water?",
     ["spoon","paper towel","plate","pen","computer"], 1),
    ("Where do you store dishes in a kitchen?",
     ["cupboard","trash can","backpack","street","bed"], 0),
    ("What do you use to cut paper?",
     ["scissors","spoon","plate","rope","glue"], 0),
    ("What do you sleep on?",
     ["table","bed","floor","sofa","chair"], 1),
    ("What do you use to write on paper?",
     ["pen","brush","crayon","ruler","eraser"], 0),
    ("What do you use to tell the time?",
     ["radio","clock","ruler","mirror","phone"], 1),
    ("What do you use to boil water?",
     ["plate","bowl","kettle","cup","bag"], 2),
    ("What do you drink when thirsty?",
     ["juice","milk","water","coffee","tea"], 2),
]

ok = 0
for question, choices, answer_idx in eval_set:
    q_fp  = pipeline.process_text(question)
    q_vec = rpenc.encode_path(q_fp)
    scores = []
    for c in choices:
        c_vec = rpenc.encode_path(pipeline.process_text(c))
        sim   = float(np.dot(q_vec, c_vec) /
                      (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9))
        scores.append(sim)
    pred    = int(np.argmax(scores))
    correct = pred == answer_idx
    ok     += int(correct)
    tag     = "PASS" if correct else "FAIL"
    print(f"  [{tag}] {question[:50]}")
    print(f"        Got: {choices[pred]!r:22s} Want: {choices[answer_idx]!r}")

print(f"\n  Raw resonance score: {ok}/{len(eval_set)}")

# ── 5. Full system QA (with rules) ───────────────────────────────────────────
print()
print("=" * 50)
print("FULL SYSTEM QA (with all features)")
print("=" * 50)

system = URCMSystem(resonance_dim=1024)
from tests.test_commonsenseqa import choose_answer

full_set = [
    ("What do people use to absorb water?",
     ["spoon","paper towel","plate","pen","computer"], 1),
    ("Where do you store dishes in a kitchen?",
     ["cupboard","trash can","backpack","street","bed"], 0),
    ("What do you use to cut paper?",
     ["scissors","spoon","plate","rope","glue"], 0),
]

ok_full = 0
for item in full_set:
    q, choices, ans_idx = item
    pred = choose_answer(system, q, choices)
    correct = pred == ans_idx
    ok_full += int(correct)
    tag = "PASS" if correct else "FAIL"
    print(f"  [{tag}] {q[:50]}")
    print(f"        Got: {choices[pred]!r:22s} Want: {choices[ans_idx]!r}")

print(f"\n  Full system score: {ok_full}/{len(full_set)}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Parameters:          {total:,}  (~{total/1e6:.1f}M)")
print(f"  Spectral radius:     {sr:.4f}")
print(f"  Hippocampus:         {hippo} entries")
print(f"  Raw resonance QA:    {ok}/{len(eval_set)}")
print(f"  Full system QA:      {ok_full}/{len(full_set)}")
if ok >= 4:
    print("  STATUS: Resonance doing real semantic work")
elif ok >= 2:
    print("  STATUS: Partial — resonance contributing, rules still needed")
else:
    print("  STATUS: Rules carrying the weight — more Hebbian training needed")
print("=" * 50)

# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np, time, warnings
warnings.filterwarnings("ignore")
from urcm.core.system import URCMSystem

FACTUAL = [
    ("What absorbs water?",       ["paper towel","spoon","plate","pen","computer"], "paper towel"),
    ("Where to store dishes?",    ["cupboard","trash can","backpack","street","bed"], "cupboard"),
    ("What cuts paper?",          ["scissors","spoon","plate","rope","glue"], "scissors"),
    ("What tells time?",          ["clock","ruler","mirror","phone","radio"], "clock"),
    ("What do you sleep on?",     ["bed","chair","table","floor","sofa"], "bed"),
    ("What boils water?",         ["kettle","bowl","cup","plate","bag"], "kettle"),
    ("What cleans teeth?",        ["toothbrush","comb","fork","spoon","brush"], "toothbrush"),
    ("What do you write with?",   ["pen","eraser","ruler","brush","crayon"], "pen"),
]

NONSENSE = [
    ("What color is the smell of Tuesday?",     ["nothing","maybe","always","never","everything"]),
    ("How heavy is a thought in kg?",           ["nothing","maybe","always","never","everything"]),
    ("What is the square root of happiness?",   ["nothing","maybe","always","never","everything"]),
    ("What temperature does silence burn at?",  ["nothing","maybe","always","never","everything"]),
    ("What is the chemical formula for love?",  ["nothing","maybe","always","never","everything"]),
    ("Where does the wind store memories?",     ["nothing","maybe","always","never","everything"]),
]

print("Initializing URCM (2048-dim, Wave Physics)...")
t0 = time.time()
system = URCMSystem(resonance_dim=2048)
print("  %.1fs\n" % (time.time() - t0))

print("=" * 65)
print("FACTUAL QA (URCM)")
print("=" * 65)
facts_ok = 0
scores_correct, scores_wrong = [], []
for q, choices, expected in FACTUAL:
    r = system.solve_qa_right_brain(q, choices)
    pred = r.get("winner", "?")
    ok = pred.lower() == expected.lower()
    score = 0.0
    for d in r.get("details", []):
        if d.get("choice", "").lower() == expected.lower():
            score = d.get("score", 0)
            break
    if ok:
        facts_ok += 1
        scores_correct.append(score)
    else:
        scores_wrong.append(score)
    tag = "PASS" if ok else "FAIL"
    print("  [%s] %-35s -> %-15s (score=%.3f)" % (tag, q, pred, score))

acc = facts_ok / len(FACTUAL)
gap = np.mean(scores_correct) - np.mean(scores_wrong) if scores_correct and scores_wrong else 0
print("  Accuracy: %d/%d = %.0f%%" % (facts_ok, len(FACTUAL), acc*100))
print("  Score gap (correct-wrong): %.3f" % gap)

print()
print("=" * 65)
print("HALLUCINATION TRAPS (Nonsense questions)")
print("=" * 65)
hallu = 0
for q, choices in NONSENSE:
    r = system.solve_qa_right_brain(q, choices)
    scores = [d.get("score", 0) for d in r.get("details", [])]
    mu = max(scores) if scores else 0
    is_hallu = mu > 0.3
    if is_hallu:
        hallu += 1
    tag = "HALLUC" if is_hallu else "SAFE"
    print("  [%s] score=%.3f -> %-15s | %s" % (tag, mu, r.get("winner","?"), q[:40]))

hallu_pct = hallu / len(NONSENSE)
print("  Hallucination rate: %d/%d = %.0f%%" % (hallu, len(NONSENSE), hallu_pct*100))

print()
print("=" * 65)
print("COMPARISON: DistilGPT2 (transformer)")
print("=" * 65)
llm_acc = 0.0
llm_hallu_pct = 1.0
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print("Loading DistilGPT2...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    print("  %.1fs" % (time.time() - t0))

    def llm_answer(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=15, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True)
        ans = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        ans = ans[len(prompt):].strip().split("\n")[0]
        probs = [torch.softmax(s, dim=-1).max().item() for s in out.scores] if out.scores else [0]
        return ans, float(np.mean(probs))

    llm_ok = 0
    for q, choices, expected in FACTUAL:
        ans, conf = llm_answer("Q: %s\nA:" % q)
        correct = expected.lower() in ans.lower()
        tag = "PASS" if correct else "FAIL"
        if correct:
            llm_ok += 1
        print("  [%s] %-35s -> %-25s (conf=%.3f)" % (tag, q, ans[:25], conf))

    llm_acc = llm_ok / len(FACTUAL)
    print("  Accuracy: %d/%d = %.0f%%" % (llm_ok, len(FACTUAL), llm_acc*100))

    llm_hallu = 0
    for q, _ in NONSENSE:
        ans, conf = llm_answer("Q: %s\nA:" % q)
        is_hallu = conf > 0.5
        if is_hallu:
            llm_hallu += 1
        tag = "HALLUC" if is_hallu else "SAFE"
        print("  [%s] conf=%.3f -> %-25s | %s" % (tag, conf, ans[:25], q[:40]))

    llm_hallu_pct = llm_hallu / len(NONSENSE)
    print("  Hallucination rate: %d/%d = %.0f%%" % (llm_hallu, len(NONSENSE), llm_hallu_pct*100))
except Exception as e:
    print("  Transformer eval skipped: %s" % e)

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print("  %-35s %10s %12s" % ("", "URCM", "Transformer"))
print("  %-35s %10s %12s" % ("Factual Accuracy", "%.0f%%" % (acc*100), "%.0f%%" % (llm_acc*100)))
print("  %-35s %10s %12s" % ("Hallucination Rate", "%.0f%%" % (hallu_pct*100), "%.0f%%" % (llm_hallu_pct*100)))
if llm_hallu_pct > hallu_pct:
    red = (llm_hallu_pct - hallu_pct) / llm_hallu_pct * 100
    print("  %-35s %10s" % ("Hallucination Reduction", "%.0f%%" % red))
print("=" * 65)

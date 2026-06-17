# -*- coding: utf-8 -*-
"""
Ablation Study: Wave Merger vs Dense ESN

Compares:
1. Full URCM with wave merger (O(B*D) dynamics)
2. Dense ESN baseline (O(D^2) dynamics, no wave compression)

Both use the same 2048-dim resonance, same training data, same evaluation.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from urcm.core.system import URCMSystem

FACTUAL = [
    ("What absorbs water?",       ["paper towel", "spoon", "plate", "pen", "computer"], "paper towel"),
    ("Where to store dishes?",    ["cupboard", "trash can", "backpack", "street", "bed"], "cupboard"),
    ("What cuts paper?",          ["scissors", "spoon", "plate", "rope", "glue"], "scissors"),
    ("What tells time?",          ["clock", "ruler", "mirror", "phone", "radio"], "clock"),
    ("What do you sleep on?",     ["bed", "chair", "table", "floor", "sofa"], "bed"),
    ("What boils water?",         ["kettle", "bowl", "cup", "plate", "bag"], "kettle"),
    ("What cleans teeth?",        ["toothbrush", "comb", "fork", "spoon", "brush"], "toothbrush"),
    ("What do you write with?",   ["pen", "eraser", "ruler", "brush", "crayon"], "pen"),
]

NONSENSE = [
    ("What color is the smell of Tuesday?",     ["nothing", "maybe", "always", "never", "everything"]),
    ("How heavy is a thought in kg?",           ["nothing", "maybe", "always", "never", "everything"]),
    ("What is the square root of happiness?",   ["nothing", "maybe", "always", "never", "everything"]),
    ("What temperature does silence burn at?",  ["nothing", "maybe", "always", "never", "everything"]),
    ("What is the chemical formula for love?",  ["nothing", "maybe", "always", "never", "everything"]),
    ("Where does the wind store memories?",     ["nothing", "maybe", "always", "never", "everything"]),
]


def run_eval(system, label):
    print("=" * 60)
    print(label)
    print("=" * 60)

    t0 = time.time()
    print("  Loading...", flush=True)

    fs, ns = [], []
    for q, ch, ex in FACTUAL:
        r = system.solve_qa_right_brain(q, ch)
        ex_sc = 0
        for d in r.get("details", []):
            if d.get("choice", "").lower() == ex.lower():
                ex_sc = d.get("score", 0)
        fs.append(ex_sc)

    for q, ch in NONSENSE:
        r = system.solve_qa_right_brain(q, ch)
        sc = [d.get("score", 0) for d in r.get("details", [])]
        ns.append(max(sc) if sc else 0)

    fs, ns = np.array(fs), np.array(ns)
    gap = fs.mean() / (ns.mean() + 1e-10)
    print("  Factual:   %.2f +/- %.2f" % (fs.mean(), fs.std()), flush=True)
    print("  Nonsense:  %.2f +/- %.2f" % (ns.mean(), ns.std()), flush=True)
    print("  Gap:       %.1fx" % gap, flush=True)
    print("  Time:      %.1fs" % (time.time() - t0), flush=True)
    return fs.mean(), ns.mean(), gap


print("Loading wave merger system...", flush=True)
s1 = URCMSystem(resonance_dim=2048, use_wave_dynamics=True)
f1, n1, g1 = run_eval(s1, "WAVE MERGER (O(B*D) dynamics)")

print(flush=True)

print("Loading dense ESN system...", flush=True)
s2 = URCMSystem(resonance_dim=2048, use_wave_dynamics=False)
f2, n2, g2 = run_eval(s2, "DENSE ESN (O(D^2) dynamics)")

print(flush=True)
print("=" * 60)
print("RESULT")
print("=" * 60)
print("  Wave gap:   %.1fx" % g1)
print("  Dense gap:  %.1fx" % g2)
if g1 > g2:
    print("  Improvement: +%.0f%%" % (100 * (g1 - g2) / (g2 + 1e-10)))
else:
    print("  No improvement")

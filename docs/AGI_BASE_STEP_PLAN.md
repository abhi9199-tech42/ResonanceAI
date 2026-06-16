# AGI Base: Step‑by‑Step Plan (Transformer‑Free Demo First)

## Goal
- Build a practical AGI base on CPU, no transformer for the demo. Use resonance reasoning, learned gating, and strong negative training. Measure μ/ρ/χ telemetry and subset accuracy, then proceed to real training.

## Prerequisites
- Python 3.10+
- Repo cloned and working in C:\Users\kriti\OneDrive\12344555
- Optional: urcm/data/urcm_training.db present with qa/synonyms/antonyms tables

## Step 1: Quick Warmup (CPU‑Only)
- Purpose: Establish initial attractors for a small set of QA seeds and repel confusers
- Command:
  - python -m urcm.cli warmup --dim 32
- Output:
  - Updates C:\Users\kriti\OneDrive\12344555\urcm_weights.pkl with W_in/W_res/W_out/bias/W_res_inv
- Time:
  - Seconds on typical CPU

## Step 2: Demo QA Subset Check
- Purpose: Validate end‑to‑end pipeline without transformer
- Command:
  - python -m urcm.cli qa --dim 32
- Target:
  - Aim for 3/3 on the demo subset (“paper towel”, “cupboard”, “scissors”)
- Telemetry:
  - For detailed metrics on a single query:
  - python -m urcm.cli query "What do people use to absorb water? paper towel" --dim 32

## Step 3: Real Training (CPU‑Only)
- Purpose: Strengthen margins with broader seeds and negatives, tune gating
- Command:
  - python urcm/tools/train_from_sqlite.py --db urcm/data/urcm_training.db --dim 32
- What it does:
  - Deposits/repels attractors using seeds + confusers
  - Hill‑climbs gate_alpha/gate_beta for last‑token gating
  - Saves urcm_weights.pkl (including gate params) and brain pickle with qa_lr_w
- Time:
  - ~5–15 minutes on CPU for 32‑dim (depends on machine/data)

## Step 4: Hardened Test Validation
- Purpose: Validate against a larger fixed set
- Command:
  - python -m pytest -q tests/test_commonsenseqa.py::test_commonsenseqa_miniset_passes -q
- Target:
  - Reduce failures; continue iterating until consistent pass

## Step 5: Iteration Loop (Error‑Driven)
- For any failed items:
  - Add failed question/answer to seeds in training
  - Add close confusers to negatives
  - Re‑run Step 3 → Step 4
- Observe telemetry trend (μ/ρ/χ) to ensure positive μ stability on correct choices

## Step 6: Optional Retrieval (Still Transformer‑Free)
- Purpose: Improve grounding for broader phrasing
- Command:
  - python urcm/tools/train_from_sqlite.py --auto --goals "Kitchen" "Household items" --dim 32
- Then craft new seeds from ingested text and repeat Steps 3–5

## Step 7: Safety & Values
- Keep SafetyGovernor engaged (energy ceiling, spectral radius bounds)
- Track metrics in system.status["metrics_history"] (μ/ρ/χ per query)
- Expand value constraints (truth/safety/benefit) in reasoning paths as needed

## Step 8: Scale Plan
- After stable demo:
  - Increase dataset size and diversify phrasing
  - Consider 64–128‑dim if CPU budget allows
  - Optionally add tiny quantized transformer for embeddings (inference‑only), keeping the decision head CPU‑based

## Quick References
- System pipeline and choice split:
  - urcm/core/system.py
- Encoder and learned final‑step gating:
  - urcm/core/resonance_encoder.py
- Training with gate tuning and negatives:
  - urcm/tools/train_from_sqlite.py
- Demo CLI:
  - urcm/cli.py

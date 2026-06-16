# URCM Gap Fix Plan

## Honest Time Estimate

| Gap | Effort | Time (solo dev) |
|---|---|---|
| BrocaArea → real retrieval decoder | Low | 2–4 hours |
| Prove resonance does semantic work (remove keyword rules) | Medium | 1–3 days |
| MeshNode — actually wire multiple nodes | Medium | 2–4 days |
| SemanticLatentSpace — task-adaptive projection | Medium | 2–3 days |
| End-to-end QA without hardcoded rules | Hard | 1–2 weeks |
| **Total realistic** | | **2–4 weeks** |

---

## Gap 1 — BrocaArea (EASY, do first)

**Problem:** 35-sentence Markov bigram. Vocabulary is ~80 words. Generates garbage for anything outside the seed corpus.

**Fix:** Replace with `ConceptDecoder` nearest-neighbor retrieval over a real vocabulary.

**What to do:**

1. Delete `urcm/core/broca.py` usage from `urcm/cli.py` (`poem`, `brainstorm` commands)
2. In `URCMSystem`, replace `compose_poem()` with a retrieval chain:
   - Encode topic → resonance vector
   - Run 4–6 dynamics hops (already done in brainstorm CLI loop)
   - At each hop, call `ConceptDecoder.decode(state, top_k=5)`
   - Pick best non-repeated concept
   - That IS the output — concept stream, not fake poetry
3. Build index from a real word list — even 5000 common English words is enough
4. Load wordlist from `training_data/` or download `words_alpha.txt` (466k English words, public domain)

**Files to change:**
- `urcm/core/broca.py` — gut `_seed_corpus`, replace `speak()` with NN retrieval
- `urcm/core/system.py` — update `compose_poem()` to use new decoder
- `urcm/cli.py` — poem/brainstorm commands already work, just need bigger vocab

**Time: 2–4 hours**

---

## Gap 2 — Keyword Rules Hiding Resonance Failure (IMPORTANT)

**Problem:** `system.py process_query()` has ~400 lines of hardcoded `if "absorb" in ql` rules. These rules are what makes CommonsenseQA pass — not the resonance vector similarity. The resonance is decoration.

**How to verify this is the problem:**
```python
# Add this test — if it still passes 2/3 with rules stripped, resonance works
# If it drops to 0/3 or 1/3, the rules are carrying all the weight
```

**Fix options (pick one):**

**Option A — Train W_res properly (2–3 days)**
- Run the warmup Hebbian deposits (already in `cli.py warmup` command) with 500+ QA pairs
- Save weights to `urcm_weights.pkl`
- Strip the keyword rules from `process_query`
- Re-run CommonsenseQA — if accuracy holds, resonance is working

**Option B — Add a thin learned scorer on top (1 day)**
- Keep resonance vectors, add a simple logistic regression on `[cosine_sim(q_vec, choice_vec), rho(choice_vec), chi(q_vec, choice_vec)]`
- Train on 200 CommonsenseQA examples
- This is `qa_lr_w` in the weights file — it's already wired, just never trained properly
- Run `python -m urcm.cli warmup --dim 1024` to populate it

**Option C — Be honest in the docs (30 min)**
- Rename `process_query` internal path to `_rule_augmented_process`
- Document that rules are a bootstrap scaffold, not the core mechanism
- This doesn't fix the gap but at least it's honest

**Recommended: Option B first (fastest), then Option A to validate**

**Files to change:**
- `urcm/core/system.py` — `process_query()`, strip/reduce rules after training
- `urcm/cli.py` — `warmup()`, extend seed QA pairs to 200+

**Time: 1–3 days**

---

## Gap 3 — MeshNode is a Stub

**Problem:** `MeshNode` in the design doc and requirements doesn't exist in the codebase. The decentralized mesh is purely spec.

**Fix:** Implement a minimal real version.

**What to build:**

```
urcm/core/mesh_node.py
  class MeshNode:
      - __init__(node_id, resonance_dim)
      - process(text) → ResonanceState
      - get_signal() → {"delta_mu": float, "phase": float}  ← no raw data
      - receive_signals(signals: List[dict]) → None  ← updates local state
      
urcm/core/mesh.py  (already exists, extend it)
  class URCMMesh:
      - add_node(node_id)
      - broadcast(text) → Dict[node_id, ResonanceState]
      - synchronize() → float  ← average order parameter
      - get_consensus() → ResonanceState  ← mean of all node states
```

**Key constraint:** `get_signal()` must never return raw vectors — only scalar Δμ and phase float. This is what makes it privacy-preserving per Requirement 5.2.

**Time: 2–4 days**

---

## Gap 4 — SemanticLatentSpace Never Adapts

**Problem:** `SemanticLatentSpace` uses a static random orthogonal projection. `task_adaptation()` is in the design but not implemented. It's a one-size-fits-all PCA-like compression.

**Fix:** Add task-conditioned projection.

```python
class SemanticLatentSpace:
    def task_adaptation(self, task_context: str) -> None:
        """
        Rotate the projection matrix toward task-relevant subspace.
        task_context: "qa", "reasoning", "retrieval", "generation"
        """
        # Simple version: learn a task-specific bias vector
        # Add it to the projection matrix with small weight
        task_biases = {
            "qa": np.load("urcm/data/task_bias_qa.npy"),
            "reasoning": np.load("urcm/data/task_bias_reasoning.npy"),
        }
        if task_context in task_biases:
            bias = task_biases[task_context]
            self.E = self.E + 0.05 * bias  # soft rotation
            # Re-orthogonalize
            Q, _ = np.linalg.qr(self.E.T)
            self.E = Q.T
            self.D = Q
```

This requires generating the task bias vectors once (offline, from example inputs).

**Time: 2–3 days**

---

## Recommended Order

```
Week 1
  Day 1-2:  Gap 2 Option B — train qa_lr_w, test without keyword rules
  Day 3:    Gap 1 — replace BrocaArea, build real vocab index
  Day 4-5:  Verify resonance is actually working, write honest benchmark

Week 2
  Day 1-3:  Gap 3 — implement real MeshNode + Mesh
  Day 4-5:  Gap 4 — task-adaptive latent space

Week 3 (if needed)
  Extend QA training data to 1000+ pairs
  Full CommonsenseQA eval without any keyword rules
  Write the "resonance does semantic work" proof test
```

---

## The One Test That Matters

Before doing any of the above, run this to know where you actually stand:

```python
# tests/test_resonance_semantic_work.py

def test_resonance_without_keyword_rules():
    """
    Strip ALL keyword rules from process_query.
    If resonance is doing real semantic work, accuracy should still be >= 2/3.
    If it drops to 0/3, the rules were carrying the weight and Gap 2 is critical.
    """
    from urcm.core.system import URCMSystem
    
    system = URCMSystem(resonance_dim=1024)
    
    # Direct vector cosine similarity — no rules, no Hebbian, raw resonance
    def raw_resonance_qa(q, choices):
        q_path = system.pipeline.process_text(q)
        q_vec = system.encoder.get_resonance_state(q_path).resonance_vector
        
        scores = []
        for c in choices:
            c_path = system.pipeline.process_text(c)
            c_vec = system.encoder.get_resonance_state(c_path).resonance_vector
            sim = float(np.dot(q_vec, c_vec) / (
                np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9
            ))
            scores.append(sim)
        return int(np.argmax(scores))
    
    dataset = [
        {"q": "What do people use to absorb water?",
         "choices": ["spoon", "paper towel", "plate", "pen", "computer"],
         "answer_idx": 1},
        {"q": "Where do you store dishes in a kitchen?",
         "choices": ["cupboard", "trash can", "backpack", "street", "bed"],
         "answer_idx": 0},
        {"q": "What do you use to cut paper?",
         "choices": ["scissors", "spoon", "plate", "rope", "glue"],
         "answer_idx": 0},
    ]
    
    ok = sum(
        int(raw_resonance_qa(d["q"], d["choices"]) == d["answer_idx"])
        for d in dataset
    )
    
    print(f"\nRaw resonance (no rules): {ok}/3")
    # Don't assert — just measure. This is diagnostic.
```

**Run this first. The result tells you everything about where the real work is.**

---

## Files Summary

| File | Action |
|---|---|
| `urcm/core/broca.py` | Replace `_seed_corpus` + `speak()` with NN retrieval |
| `urcm/core/system.py` | Strip/reduce keyword rules after Hebbian training |
| `urcm/core/latent_space.py` | Add `task_adaptation()` implementation |
| `urcm/core/mesh.py` | Extend with real multi-node sync |
| `urcm/core/mesh_node.py` | Create — real `MeshNode` with signal-only comms |
| `urcm/cli.py` | Extend `warmup()` with 200+ QA seeds |
| `tests/test_resonance_semantic_work.py` | Create — diagnostic baseline test |
| `training_data/` | Add `words_alpha.txt` or similar vocab file |

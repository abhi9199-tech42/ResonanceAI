# Training Guide: Teach ResonanceAI Your Domain

Train ResonanceAI on custom domains — drones, robot arms, autonomous vehicles, voice assistants, or any structured QA task.

---

## Dataset Format

A training dataset is a JSON list of triples:

```json
[
  [
    "question text",
    "correct answer",
    ["wrong answer 1", "wrong answer 2", "wrong answer 3", "wrong answer 4"]
  ]
]
```

Each triple must have exactly 1 correct answer and at least 3 wrong alternatives.

### Example (household)

```json
[
  ["What do you use to cut paper?", "scissors", ["spoon", "plate", "glue", "rope"]],
  ["What do you sleep on?",         "bed",      ["chair", "table", "floor", "sofa"]]
]
```

### Example (drone obstacle avoidance)

```json
[
  ["Is this obstacle safe to pass?",  "cardboard box",  ["concrete wall", "chain-link fence", "power line", "glass window"]],
  ["Is this landing surface stable?", "flat grass",     ["water", "gravel", "sloped roof", "mud"]],
  ["Is this object avoidable?",       "tree branch",    ["building", "mountain", "another drone", "crowd"]]
]
```

### Example (robot grasping)

```json
[
  ["Can I grasp this object?",           "mug with handle",     ["heavy box", "slippery pipe", "tiny screw", "liquid container"]],
  ["Is this object pick-and-place safe?","lightweight cube",    ["sharp blade", "fragile glass", "irregular rock", "cable"]],
  ["What is the orientation of this?",   "upright",             ["upside-down", "tilted 45deg", "horizontal", "on its side"]]
]
```

### Example (autonomous vehicle)

```json
[
  ["What color is the traffic light?",    "green",        ["red", "yellow", "flashing", "off"]],
  ["Is this pedestrian crossing safe?",   "stopped, waiting", ["running across", "hidden behind car", "with bike", "child running"]],
  ["Is this lane change safe?",           "clear, no cars",   ["car blind spot", "fast approaching", "motorcycle", "bicycle"]]
]
```

---

## Recommended Dataset Sizes

| Task | Minimum Pairs | Recommended | Expected AUROC |
|------|--------------|-------------|----------------|
| Household QA | 30 | 62 (included) | 1.0 |
| Drone obstacle | 50 | 200+ | 0.85–0.95 |
| Robot grasping | 50 | 300+ | 0.85–0.95 |
| Autonomous vehicle | 100 | 500+ | 0.80–0.95 |
| Voice assistant | 100 | 500+ | 0.85–0.95 |
| IoT sensor alerts | 30 | 100+ | 0.90–0.95 |

More pairs = better discrimination between similar concepts.  
Minimum 3 wrong alternatives per question recommended.

---

## How to Train

### 1. Prepare your dataset

Create a JSON file (e.g., `my_drone_pairs.json`) with the format above.

### 2. Run training

```bash
python train_2048.py  # pass your JSON pairs file path
```

This will:
1. Initialize base weights (W_in, W_res, W_out, bias)
2. Run Hebbian deposits for each QA pair (800 cycles per pair)
3. Repel wrong answers (300 cycles per wrong answer)
4. Restore rank + spectral clip (prevents attractor collapse)
5. Train a logistic QA scorer on the feature vector
6. Re-encode hippocampus entries with final weights
7. Save to `urcm_weights.pkl`

Training time: ~55s for 62 pairs. Scales linearly with pair count.

### 3. Evaluate

```bash
python -c "
from urcm.core.system import URCMSystem
system = URCMSystem(resonance_dim=2048)

# Quick sanity check
for question, correct, wrongs in my_test_pairs[:5]:
    result = system.verify_qa(question, correct)
    print(f'{question}: {correct} -> conf={result[\"confidence\"]:.3f}')
"
```

### 4. Full benchmark

```bash
python tests/production/run.py --report
```

---

## Tips for Good Datasets

### Wrong answers matter

Bad wrong answers → inflated AUROC.  
Good wrong answers → realistic discrimination.

```
❌ Weak:   ["spoon", "plate", "pen", "computer"]
✅ Strong: ["spoon", "plate", "glue", "rope"]
```

The wrong answers should be **plausible but incorrect** — concepts the system might confuse.

### Cover edge cases

Include questions where:
- Two answers are very similar ("mug" vs "cup")
- The answer depends on context ("safe obstacle" vs "unsafe obstacle")
- Negative questions ("What should you NOT use to cut paper?")

### Balance domains

For a drone: mix obstacle type, weather condition, landing surface, altitude decision, battery level questions. Don't overfit to one sub-domain.

---

## One-Shot Learning (Runtime)

For adding new concepts without retraining:

```python
system.learn_concept_oneshot(
    concept="new_obstacle_type",
    definition="a low-hanging tree branch at 2 meters height, safe to pass under"
)
```

Limitations:
- Only ~50 Hebbian cycles (vs 800 in batch training)
- No logistic scorer update
- Good for 10-20 additions before batch retrain is needed

For production: batch retrain every 100 new concepts.

---

## Pipeline Overview

```
Raw data (JSON pairs)
    ↓
train_2048.py
    ├── init_base_weights()     → W_in, W_res, W_out, bias
    ├── train_hebbian()         → W_res + hippocampus
    │   ├── shock_deposit (800 cycles)   Q→A, A→Q
    │   ├── repel wrong answers (300 cy)
    │   └── rank restoration + spectral clip
    ├── train_qa_scorer()       → logistic regression weights
    ├── re-encode hippocampus   → final vectors
    └── save to urcm_weights.pkl
    ↓
URCMSystem(resonance_dim=2048)
    ├── detect_hallucination(text)       → confidence (text-only)
    ├── verify_qa(question, answer)      → confidence (question-aware)
    └── learn_concept_oneshot()          → runtime learning
```

---

## Domain-Specific Dataset Recommendations

### Drone / UAV

| Source | Notes |
|--------|-------|
| AirSim logs | Simulate obstacle scenarios, extract QA |
| Real flight logs | Tag successful vs failed maneuvers |
| Synthetic generation | Use LLM to generate (question, safe, unsafe) triples from obstacle descriptions |

**Recommended pairs**: 200+ covering obstacle type, altitude, weather, battery, landing surface.

### Robot Arm / Manipulation

| Source | Notes |
|--------|-------|
| Grasp success logs | Record object type + grasp outcome |
| Simulation (Isaac Gym, MuJoCo) | Generate varied object/scenario QA |
| Human demonstration | Tag graspable vs non-graspable objects |

**Recommended pairs**: 300+ covering object shape, weight, orientation, surface texture.

### Autonomous Vehicle

| Source | Notes |
|--------|-------|
| Public datasets (BDD100K, nuScenes) | Extract scene QA from annotations |
| Driving log review | Tag safe vs unsafe decisions |
| Simulation (CARLA) | Generate edge-case scenarios |

**Recommended pairs**: 500+ covering traffic lights, pedestrians, lane changes, weather.

### Voice Assistant

| Source | Notes |
|--------|-------|
| User query logs (anonymized) | Extract question-answer pairs |
| FAQ documents | Structured Q&A from product docs |
| Synthetic generation | LLM-generated QA pairs for your domain |

**Recommended pairs**: 500+ covering your specific domain (home, medical, industrial).

---

## FAQ

**Q: Can I mix domains in one training run?**  
A: Yes. Mix drone + robot + vehicle pairs. The hippocampus handles all of them. But more diverse domains need more total pairs.

**Q: How long does training take?**  
A: ~55s for 62 pairs. ~5min for 500 pairs. ~15min for 2000 pairs. O(n) in pair count.

**Q: Can I train incrementally?**  
A: Batch training only (full retrain). For small additions, use `learn_concept_oneshot()` at runtime.

**Q: What if my answers are longer than 3 words?**  
A: ResonanceAI is optimized for short answers (1-3 words). Long answers may get lower AUROC. Consider splitting long answers into structured sub-questions.

**Q: Can I use this on a Raspberry Pi?**  
A: Yes. 32MB model, CPU-only, 0.2s per query. Tested on RPi 4.

---

## Quick Reference

```bash
# Train
python train_2048.py  # pass your JSON pairs file path

# Evaluate
python tests/production/run.py --report

# CLI inference
resonanceai qa "Is this safe?" --choices "yes,no,unsure"
```

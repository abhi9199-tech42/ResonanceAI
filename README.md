# ResonanceAI

A reasoning engine that processes language as frequency signals instead of tokens. It uses wave compression to run faster than standard recurrent networks and rejects questions it doesn't know rather than making up answers.

---

## What It Does

- Converts text to phoneme-based frequency vectors
- Runs dynamics using compressed echo state networks (32 frequency bands)
- Scores inputs based on how well they match learned patterns
- Low scores = "I don't know" — high scores = confident answer

---

## Results

Tested on 8 simple factual questions and 6 nonsense questions:

| What | Result |
|------|--------|
| Factual accuracy | 6/8 correct (75%) |
| Hallucination on nonsense | 0/6 (0%) |
| Score range for known answers | 2 – 6,453 |
| Score range for nonsense | 1.2 – 14.2 |

The gap between known and unknown is large. A threshold of 0.3 separates them cleanly.

**What this does not mean:**
- This is not tested on real-world tasks
- 8 questions is not a benchmark
- No adversarial or out-of-domain testing
- Trained on only 62 QA pairs

---

## Speed

Standard echo state networks do `state @ W_res` — a matrix multiply of size D×D.

The wave merger breaks this into 3 steps using 32 frequency bands:

1. Project state into 32 bands: O(32 × D)
2. Evolve in band space: O(32²)
3. Project back: O(32 × D)

At D=2048: 131K operations instead of 4.1M. That's 31× fewer operations.

The 32 bands are a fixed cosine grid, not learned. The coupling between bands (32×32 matrix) is learned.

---

## How to Use

```bash
pip install -r requirements.txt

# Run tests
python -m pytest

# Quick test
python -c "
from urcm.core.system import URCMSystem
s = URCMSystem(resonance_dim=2048)
r = s.process_query('What do you use to cut paper?')
print('Converged:', r.convergence_achieved)
"

# Run hallucination benchmark
python hallucination_benchmark.py

# Train new weights
python train_2048.py
```

---

## Architecture

```
Text
 → PhonemeMapper (char to phoneme to 24-dim vector)
 → ResonanceEncoder (24 → 2048 dim, echo state network)
   → WaveMerger (32-band compression, O(B·D) per step)
   → OscillatoryGating (tanh × sigmoid)
   → AttractorNetwork (phase synchronization)
   → MuConvergence (stop when stable)
 → Memory (Hebbian deposits, one-shot learning)
 → Output (nearest neighbor or Markov decoder)
```

---

## Weight Files

| File | Shape | What |
|------|-------|------|
| W_in | 24 × 2048 | Input projection |
| W_res | 2048 × 2048 | Recurrent weights |
| W_out | 2048 × 24 | Decoder |
| qa_lr_w | 5 | QA scorer weights |
| hippocampus | 124 entries | Memory |

---

## Files

```
urcm/
├── core/
│   ├── wave_merger.py          # Wave compression
│   ├── resonance_encoder.py    # Echo state network
│   ├── phoneme_mapper.py       # Text to frequency
│   ├── system.py               # Main system
│   ├── memory.py               # Hebbian memory
│   └── ...
├── tests/                      # 140+ tests
├── train_2048.py               # Training
├── hallucination_benchmark.py  # Evaluation
└── requirements.txt
```

---

## Merge with Transformers

See [URCM_TRANSFORMER_MERGE_GUIDE.md](./URCM_TRANSFORMER_MERGE_GUIDE.md) for how to integrate with HuggingFace, PyTorch, or JAX.

---

## License

Apache 2.0

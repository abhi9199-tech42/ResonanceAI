# ResonanceAI Technical Documentation

## 1. What This System Does

ResonanceAI is a reasoning engine that processes language as frequency signals instead of predicting the next token. It converts text to phoneme-based frequency vectors, runs them through an echo state network with wave compression, and scores the output based on how well it matches learned patterns.

The key metric is μ (mu): the ratio of semantic density (ρ) to transformation cost (χ). High μ means clear, low-effort meaning. Low μ means the system is guessing.

---

## 2. How It Works

1. Input text → broken into phonemes → converted to 24-dim frequency vectors
2. Frequency vectors → projected into 2048-dim resonance state via W_in
3. Resonance state → evolved through dynamics steps using wave-compressed W_res
4. At each step, μ is computed: ρ / (χ + ε)
5. When μ stops changing (Δμ < ε), the system stops
6. Final state is decoded back to text or scored

---

## 3. Components

### 3.1 URCMSystem (`urcm.core.system`)

Main coordinator. Initializes all subsystems and runs queries.

```python
from urcm.core.system import URCMSystem

s = URCMSystem(resonance_dim=2048)
result = s.process_query("What do you use to cut paper?")
```

### 3.2 PhonemeMapper (`urcm.core.phoneme_mapper`)

Converts text to frequency vectors:
- Breaks text into phonemes (e.g., "cat" → /k/ /æ/ /t/)
- Maps each phoneme to a base frequency
- Combines into a single 24-dim vector

### 3.3 ResonanceEncoder (`urcm.core.resonance_encoder`)

Runs the echo state network dynamics:
- `state = tanh(input @ W_in + prev_state @ W_res + bias)`
- With wave compression: decompose into 32 bands, evolve in band space, reconstruct

### 3.4 WavePhysicsMerger (`urcm.core.wave_merger`)

Compresses the O(D²) matrix multiply into O(B·D):
- Decompose state into 32 frequency bands
- Apply W_res in band space (32×32 matrix)
- Reconstruct full state

### 3.5 MuConvergenceEngine (`urcm.core.convergence_engine`)

Decides when to stop thinking:
- Maintains a beam of reasoning paths (default width: 3)
- Prunes paths with low μ
- Stops when μ stabilizes

**The μ-metric:**
- ρ (density) = 1 - normalized entropy of state → how concentrated the information is
- χ (cost) = L2 distance from previous state → how much the state changed
- μ = ρ / (χ + ε) → high = clear meaning, low cost

### 3.6 AttractorNetwork (`urcm.core.attractor_network`)

Phase synchronization using Hopfield-Kuramoto dynamics:
- Multiple states are pulled toward a shared phase
- Helps stabilize the resonance state

### 3.7 GeometricMemory (`urcm.core.memory`)

One-shot learning via Hebbian rank-1 deposits:
- `W += outer(key, arctanh(value) - key @ W) / |key|²`
- No backpropagation needed
- Capacity: ~resonance_dim × 0.5 deposits before interference

### 3.8 OscillatoryGating (`urcm.core.oscillatory_gating`)

Phase-modulated activation:
- `output = tanh(state) × sigmoid(sin(φ) + cos(φ))`
- Adds temporal structure to the dynamics

---

## 4. Weight Matrices

| Matrix | Shape | Init | Purpose |
|--------|-------|------|---------|
| W_in | 24 × 2048 | Random normal | Projects frequency vectors into resonance space |
| W_res | 2048 × 2048 | Orthogonal × 0.95 | Recurrent dynamics |
| W_out | 2048 × 24 | Pseudoinverse of W_in | Decodes resonance state back to input space |
| bias | 2048 | Small random | Offset |

After training, W_res has 124 Hebbian deposits from 62 QA pairs.

---

## 5. Complexity

Standard echo state: O(D²) per step = 4.1M operations at D=2048.

Wave-compressed: O(B·D + B²) per step = 131K operations at D=2048, B=32.

That's 31× fewer operations per step.

---

## 6. Safety

The SafetyGovernor (`urcm.core.safety`) does:
- Input clipping: prevents extreme values from entering W_in
- Energy ceiling: clamps hidden states to prevent blow-up
- Kernel lock: prevents runtime modification of core weights

---

## License

Apache 2.0

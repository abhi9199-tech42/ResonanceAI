# How to Add ResonanceAI to a Transformer

This guide shows how to plug ResonanceAI's frequency-based reasoning into an existing Transformer model.

---

## What ResonanceAI Adds

- **Hallucination rejection**: Scores nonsense inputs low, so you can refuse to answer
- **Online learning**: One-shot memory deposits without backpropagation
- **Fast dynamics**: 32-band wave compression instead of full matrix multiply
- **Grounded input**: Phoneme-based frequency vectors instead of learned embeddings

---

## Three Ways to Integrate

### 1. Replace Token Embeddings (Simplest)

Swap `nn.Embedding` with ResonanceAI's phoneme-to-frequency pipeline.

```python
import torch.nn as nn
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
import numpy as np
import pickle

class ResonanceEmbedding(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
        with open("urcm_weights.pkl", "rb") as f:
            w = pickle.load(f)
        self.W_in = nn.Parameter(torch.from_numpy(w["W_in"].T.astype(np.float32)))
        self.proj = nn.Linear(2048, d_model)

    def forward(self, texts):
        embeddings = []
        for text in texts:
            path = self.pipeline.process_text(text)
            freq = torch.from_numpy(path.vectors.astype(np.float32))
            res = freq @ self.W_in.T
            embeddings.append(self.proj(res))
        return torch.stack(embeddings)
```

When to use: You want grounded input representations without changing your model architecture.

---

### 2. Add a Resonance Layer (Middle)

Put ResonanceAI between your encoder and decoder. The encoder output goes through wave dynamics until stable, then feeds the decoder.

```python
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.wave_merger import WavePhysicsMerger

class ResonanceLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = ResonancePathEncoder(
            input_dim=d_model, resonance_dim=2048,
            use_wave_dynamics=True
        )
        self.proj_in = nn.Linear(d_model, 2048)
        self.proj_out = nn.Linear(2048, d_model)

    def forward(self, x):
        z = self.proj_in(x.mean(dim=1)).detach().numpy()
        stable_states = []
        for i in range(z.shape[0]):
            state = z[i]
            for _ in range(20):
                state = self.encoder.wave.wave_step(state)
            stable_states.append(state)
        z_stable = torch.from_numpy(np.stack(stable_states))
        return self.proj_out(z_stable)
```

When to use: You want stability and hallucination rejection as a plug-in layer.

---

### 3. Replace KV Cache with Memory (Advanced)

Replace attention's key-value cache with ResonanceAI's Hebbian memory. The cache doesn't grow — it's always 2048×2048.

```python
class ResonanceAttention(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        with open("urcm_weights.pkl", "rb") as f:
            w = pickle.load(f)
        self.W_res = nn.Parameter(torch.from_numpy(w["W_res"].astype(np.float32)))
        self.merger = WavePhysicsMerger(resonance_dim=dim, num_bands=32)

    def write(self, key_vec, value_vec):
        u = key_vec.detach().numpy()
        v = value_vec.detach().numpy()
        W = self.W_res.detach().numpy()
        error = np.arctanh(np.clip(v, -0.999, 0.999)) - u @ W
        delta = np.outer(u, error) / (np.dot(u, u) + 1e-10)
        self.W_res.data = torch.from_numpy(W + delta)

    def read(self, query, steps=10):
        state = query.detach().numpy()
        for _ in range(steps):
            state = self.merger.wave_step(state)
        return torch.from_numpy(state)
```

When to use: You need online learning or long context without growing memory.

---

## HuggingFace Example

```python
from transformers import LlamaForCausalLM

class ResonanceLlama(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.resonance_layer = ResonanceLayer(config.hidden_size)

    def forward(self, input_ids, **kwargs):
        outputs = self.model(input_ids, output_hidden_states=True, **kwargs)
        hidden = outputs.hidden_states[-1].mean(dim=1)
        stabilized = self.resonance_layer(hidden)
        return super().forward(inputs_embeds=stabilized.unsqueeze(1), **kwargs)
```

---

## Hallucination Check (No Architecture Change)

Score your model's output with ResonanceAI. If the score is low, refuse the answer.

```python
from urcm.core.system import URCMSystem

system = URCMSystem(resonance_dim=2048)
score = system.solve_qa_right_brain(question, answer)

if score < 0.3:
    print("Low confidence — might be hallucinated")
else:
    print(f"Score: {score} — likely correct")
```

---

## μ-Metric as Training Loss

Add this loss term during fine-tuning to encourage stable representations:

```python
def resonance_loss(state, prev_state):
    rho = 1 - torch.exp(-torch.var(state))
    chi = torch.norm(state - prev_state, dim=-1)
    mu = rho / (chi + 1e-8)
    return 1.0 - mu.mean()

# In training loop:
total_loss = lm_loss + 0.01 * resonance_loss(hidden[-1], hidden[-2])
```

---

## Migration Steps

1. Install: `pip install -r requirements.txt`
2. Add hallucination check (no arch change, 10 minutes)
3. Replace embeddings with ResonanceEmbedding (1 day)
4. Add ResonanceLayer between encoder/decoder (2 days)
5. Switch to ResonanceAttention if you need online learning (1 week)
6. Add μ-loss during fine-tuning (optional)

---

## Common Problems

**W_res gets unstable after many deposits**
Run spectral clipping every 50 deposits:
```python
from urcm.core.memory_maintenance import spectral_clip
W_res = spectral_clip(W_res, max_sigma=1.5)
```

**Memory search is slow with many entries**
Use FAISS for approximate nearest neighbor:
```python
import faiss
index = faiss.IndexFlatIP(2048)
index.add(hippocampus_vectors)
```

**Phoneme pipeline drops unknown characters**
The character-to-phoneme map only handles English. For other languages, extend the map or use a G2P library.

---

## File Reference

| File | What It Does |
|------|-------------|
| wave_merger.py | Wave compression (32-band decomposition) |
| resonance_encoder.py | Echo state network with wave dynamics |
| phoneme_mapper.py | Text to frequency vectors |
| system.py | Main orchestrator |
| memory.py | Hebbian one-shot memory |
| memory_maintenance.py | Keep W_res stable |
| theory.py | μ/ρ/χ math |
| safety.py | Energy clamping |

---

## License

Apache 2.0

# ResonanceAI Usage

## Running

```bash
# Install
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

# Hallucination benchmark
python hallucination_benchmark.py

# Train new weights
python train_2048.py
```

## Metrics

The system tracks μ, ρ, χ for each query in `URCMSystem.status["metrics_history"]`.

## Limits

- Trained on 62 QA pairs — not a general-purpose model
- English only (phoneme pipeline is English-focused)
- No pretrained transformer integration
- Memory capacity limited to ~1024 deposits before interference

## Tests

140+ tests covering encoder output, metrics, and basic regression.

---

Apache 2.0

# URCM Usage and Limits

## Usage
- Train: `python urcm/tools/train_from_sqlite.py --db urcm/data/urcm_training.db --dim 32`
- Query: `python -m urcm.cli query "your text" --dim 32`
- QA subset: `python -m urcm.cli qa --dim 32`

## Limits
- Reasoning relies on reservoir dynamics; generalization is limited without broad training.
- No pretrained transformer integration; transformer encoder is a stub.
- CommonsenseQA hardened test may fail; continue strengthening training and encoder gating.

## Metrics
- System records μ, ρ, χ for each query in `URCMSystem.status["metrics_history"]`.

## Tests and CI
- Basic regression tests cover encoder output dimension and metrics capture.
- CI compiles modules and runs basic tests on Windows.

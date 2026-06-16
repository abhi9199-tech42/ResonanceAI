# URCM vs Transformers: Scale, Long‑Context, and Benchmarking Plan

## Objectives
- Demonstrate competitive performance with small Transformer baselines on QA/NLI/generation.
- Improve long‑context synthesis via oscillatory gating and dynamics tuning.
- Expand knowledge coverage using curated deposition and corpus training.
- Provide transparent, reproducible measurements with energy/telemetry traces.

## Workstreams

### WS1 — Scale & Coverage
- Data
  - Curate seed QA pairs and synonyms/antonyms in SQLite at `urcm/data/urcm_training.db`.
  - Add general text corpora (Wikipedia subset, Books/CC extracts) for Hebbian training.
- Methods
  - Tabula rasa or low‑init training using tools:
    - `python -m urcm.tools.train_massive --file <path>`
    - `python -m urcm.tools.train_from_sqlite --db urcm/data/urcm_training.db`
  - Use confuser repulsion and synonym attraction during deposition.
  - Periodic checkpoints of `W_res`, maintain inverse/caches for fast retrieval.
- Measurements
  - Vocabulary growth, nearest‑neighbor diversity.
  - Retrieval accuracy on held‑out QA seeds.
  - Margin (positive vs negative cosine) over time.

### WS2 — Long‑Context Synthesis
- Experiments
  - Synthetic long sequences: story QA, multi‑hop chains, summarization prompts.
  - Scale context length and measure stability/accuracy.
- Tuning
  - Oscillatory gating parameters and temperature scheduling.
  - Metacognitive shocks to escape high‑energy plateaus.
  - Latent drift validation and phoneme‑region projection.
- Instruments
  - Track energy/stability, epiphany rate, and convergence steps.
  - Compare “right‑brain” resonance vs analytical pipeline on the same tasks.

### WS3 — Benchmarked Parity
- Datasets (lightweight subsets first)
  - CommonsenseQA (subset), BoolQ (subset), COPA, StoryCloze (val), PIQA.
- Baselines
  - DistilBERT/MiniLM or similar compact Transformer checkpoints.
  - Equalize training budget and input data where appropriate.
- Protocols
  - Zero‑shot, few‑shot (seed deposition), and tuned settings.
  - Report accuracy and statistical confidence; include ablations.

## Metrics & Reporting
- Core
  - Accuracy on benchmark subsets.
  - Positive–negative margin for QA choices.
  - Energy traces (final energy, stability variance), epiphany rate.
- Ablations
  - With/without confuser repulsion.
  - Gating enabled vs disabled.
  - Shock parameters and dynamics step caps.
- Significance
  - Bootstrap confidence intervals over question sets.

## Tooling & Commands
- Warmup seeds and confuser repulsion
  - `python -m urcm.cli warmup --dim 32`
- QA evaluation (analytical)
  - `python -m urcm.cli qa --dim 32`
- QA evaluation (right‑brain with telemetry)
  - `python -m urcm.cli qa_rb --dim 32 --show_telemetry`
- Right‑brain dynamics probe
  - `python -m urcm.cli rb "your prompt" --dim 32 --steps 50`
- Stress test
  - `python -m urcm.cli stress --dim 32 --duration 60`
- Training
  - `python -m urcm.tools.train_massive --file <corpus.txt>`
  - `python -m urcm.tools.train_from_sqlite --db urcm/data/urcm_training.db`

## Milestones
1. Prep baselines and subset datasets; verify CLI runs end‑to‑end.
2. Coverage pass with Hebbian training; track margin improvements and vocab growth.
3. Long‑context tuning: gating schedules, shock limits, stability thresholds.
4. Benchmark runs + ablations; collect telemetry and CI on results.
5. Consolidate scripts, documentation, and reproducibility artifacts.

## Risks & Mitigations
- Frequency drift / semantic collapse
  - Use latent validation and phoneme‑region projection; enable recovery routines.
- Oscillation desync
  - Phase reset and gating parameter bounds.
- Overfitting to confusers
  - Maintain held‑out confuser sets; rotate negative samples.
- Dimension mismatches
  - Enforce dimension guards and re‑init policies before long runs.

## Acceptance Criteria
- QA subset accuracy meets or exceeds compact Transformer baselines, or URCM shows superior margin/telemetry on curated tasks.
- Long‑context tasks exhibit stable energy patterns and non‑degrading accuracy at increased lengths.
- All results are reproducible with documented commands and fixed seeds.

## Resources
- Hardware: CPU acceptable for prototypes; GPU optional for larger corpora.
- Storage: checkpoints for `W_res` and SQLite training DB.
- Dependencies: Python 3.10+, NumPy, SQLite; ensure consistent environment.


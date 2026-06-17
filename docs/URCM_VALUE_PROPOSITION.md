# ResonanceAI — What It Does and Why

## The Problem

Large language models generate text by predicting the next word. They don't know if what they're saying is true — they just know if it sounds right. When they don't know the answer, they guess instead of saying "I don't know."

This is called hallucination, and it's a real problem for any application where wrong answers matter.

## What ResonanceAI Does Differently

ResonanceAI doesn't predict the next token. It converts text to frequency vectors and runs them through a dynamics system that either stabilizes (high μ score) or doesn't (low μ score).

- High score = the system recognizes this pattern
- Low score = the system doesn't recognize this — probably wrong

This gives you a built-in rejection mechanism: refuse to answer when the score is low, rather than guessing.

## Current State

- Trained on 62 commonsense QA pairs (proof of concept)
- 75% accuracy on 8 test questions
- 0% hallucination on 6 nonsense questions
- 31× faster dynamics via wave compression

## What It Doesn't Do

- Not a general-purpose language model
- Not tested on real-world benchmarks
- No knowledge base beyond 62 training pairs
- No multilingual support

## Where It Might Be Useful

- Applications where wrong answers are worse than no answer
- Systems that need to know when they don't know something
- Low-resource environments where large models are too slow
- Online learning scenarios where you need to add knowledge without retraining

## Technical Summary

| Feature | ResonanceAI | Standard LLM |
|---------|------------|--------------|
| What it does | Scores input against learned patterns | Predicts next token |
| Hallucination | Rejects unknown inputs | Guesses |
| Training | Hebbian one-shot deposits | Backpropagation |
| Dynamics | O(B·D) wave compression | O(D²) matrix multiply |
| Memory | Fixed-size (2048×2048) | Grows with context |

---

Apache 2.0

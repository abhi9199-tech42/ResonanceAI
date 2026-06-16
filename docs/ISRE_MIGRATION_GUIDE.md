# ISRE to URCM Migration & Import Guide

Since the ISRE (Intentional Semantic Reasoning Engine) codebase is not currently accessible in the workspace, this guide clearly defines **what** should be imported and **why**, based on the architectural requirements of URCM.

## 1. High-Priority Imports (The "Mind" of the System)

URCM provides the *mechanism* ($\mu$-Resonance), but it needs the *content* and *intent* structures from ISRE.

### A. Intent Graphs (`isre/core/intent_data_models.py` or similar)
*   **Why**: URCM resonates on *meanings*, but we need a structure to define what an "Intent" looks like.
*   **Action**: Import `IntentNode`, `GoalHierarchy` or equivalent classes.
*   **Destination**: `urcm/integration/intent_models.py`

### B. Context & Knowledge Base (`isre/knowledge/` or `isre/context_manager.py`)
*   **Why**: Resonance requires a medium to resonate *against*. The "Attractor States" in URCM need to be initialized from existing knowledge, not just random vectors.
*   **Action**: Import the knowledge storage or context window management logic.
*   **Destination**: `urcm/core/context_loader.py`

### C. Scenarios & Datasets (`isre/data/scenarios/`)
*   **Why**: You mentioned "WW2 Strategy Simulation" in previous sessions. URCM needs complex, ambiguous scenarios to prove it is better than standard logic.
*   **Action**: Import any JSON/Text scenario files.
*   **Test Case**: We will run a `PhonemeFrequencyPipeline` on these scenarios to see if URCM allows for "fuzzier" but more robust strategic matching than ISRE's previous engine.
*   **Destination**: `tests/data/isre_scenarios/`

## 2. Methodology: How to Import

Since the ISRE folder is protected, please perform **one** of the following:

### Option A: Add Workspace (Recommended)
1.  Go to your IDE/Editor settings.
2.  Add `C:\Users\kriti\OneDrive\Intentional Semantic Reasoning Engine (ISRE)` to the active workspace.
3.  Tell me: *"I have added the workspace."*
4.  result: I will automatically scan and copy the dependencies for you.

### Option B: Manual Copy
Copy the folders manually into `urcm/external/isre_core/` and let me know.

## 3. Integration Interface

Once imported, we will wrap ISRE components in URCM Resonance Containers:

```python
# urcm/integration/bridge.py
from urcm.core.theory import URCMTheory
from urcm.external.isre_core import IntentNode

def resonate_on_intent(intent: IntentNode) -> float:
    # Convert ISRE Intent -> URCM Phoneme Vector
    # Measure Resonance (mu)
    return URCMTheory.compute_mu(...)
```

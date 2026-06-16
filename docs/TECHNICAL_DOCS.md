# URCM Technical Documentation

**Unified μ-Resonance Cognitive Mesh (URCM)**

## 1. System Overview

The Unified μ-Resonance Cognitive Mesh (URCM) is a continuous, frequency-based reasoning architecture that replaces discrete token prediction with resonance-based state evolution. 

Unlike traditional LLMs which predict the next probable token, URCM evolves a "semantic waveform" until it stabilizes into a "truth" state. This stability is mathematically defined by the **$\mu$-Resonance** metric.

### Core Philosophy
- **Continuous Semantics:** Meaning is a high-dimensional oscillation, not a static vector.
- **Reasoning as Convergence:** Thinking is the process of a chaotic system settling into an attractor state.
- **Intrinsic Truth:** Validity is measured by constructive interference ($\mu$) with the knowledge base.

---

## 2. Architecture

The system is organized into a pipeline that transforms raw text into a converged reasoning path.

### High-Level Data Flow

1.  **Input:** Text String ("What is 2+2?")
2.  **Transduction:** Text $\rightarrow$ Phonemes $\rightarrow$ Frequency Superposition Vector ($V_0$)
3.  **Evolution:** $V_t$ evolves via differential equations (Resonance Encoder)
4.  **Competition:** Multiple paths compete for stability (Convergence Engine)
5.  **Attraction:** Paths are pulled toward stable memories (Attractor Network)
6.  **Convergence:** The system settles when $\Delta \mu \approx 0$
7.  **Output:** The final stable state is decoded back to text.

---

## 3. Core Components

### 3.1 URCMSystem (`urcm.core.system`)
The central coordinator. It initializes all subsystems and manages the lifecycle of a query.

**Key Responsibilities:**
- Integration of the pipeline.
- State management.
- Error recovery orchestration.

```python
from urcm.core.system import URCMSystem

urcm = URCMSystem(frequency_dim=24, resonance_dim=64)
result = urcm.process_query("Hello world")
```

### 3.2 Phoneme Frequency Pipeline (`urcm.core.phoneme_mapper`)
Converts discrete text into continuous signal space.

- **Tokenization:** Breaks text into phonemes (e.g., "cat" -> /k/ /æ/ /t/).
- **Frequency Mapping:** Maps each phoneme to a specific base frequency.
- **Superposition:** Combines frequencies into a single initial state vector.

### 3.3 Resonance Encoder (`urcm.core.resonance_encoder`)
The "engine" of the vehicle. It applies the system dynamics to evolve the state vector from time $t$ to $t+1$.

- **Dynamics:** $\frac{dV}{dt} = F(V, t)$
- **Role:** Generates the "next thought" candidate based on the current trajectory.

### 3.4 Mu Convergence Engine (`urcm.core.convergence_engine`)
The "driver" or "critic". It evaluates generated states and decides which reasoning path to pursue.

**The $\mu$-Metric:**
$$ \mu = \frac{\rho}{\chi + \epsilon} $$

- **$\rho$ (Rho - Density):** Information clarity (Inverse Entropy). High $\rho$ = Clear meaning.
- **$\chi$ (Chi - Cost):** Cognitive effort (Displacement). High $\chi$ = Forced logic/Hallucination.

**Logic:**
- Maintains a **Beam** of active reasoning paths (default width: 3).
- Prunes paths with low $\mu$.
- Declares success when a path stabilizes ($\Delta \mu < \epsilon$).

### 3.5 Attractor Network (`urcm.core.attractor_network`)
The Long-Term Memory (LTM).

- **Structure:** A manifold with "basins of attraction".
- **Function:** As the reasoning vector passes near a known concept (Attractor), it is naturally pulled towards it. This ensures the system defaults to known truths rather than drifting into noise.

### 3.6 Semantic Latent Space (`urcm.core.latent_space`)
A compressed representation of the resonance states, used for efficient storage and reconstruction.

---

## 4. Data Models

### ResonanceState
Represents a single moment in the cognitive process.
- `resonance_vector`: The raw $N$-dimensional state.
- `mu_value`: Stability score.
- `rho_density`: Semantic density.
- `chi_cost`: Transformation cost.
- `timestamp`: Logic time.

### ReasoningPath
A history of states representing a complete train of thought.
- `states`: List of `ResonanceState`.
- `total_energy`: Cumulative cognitive effort.
- `converged`: Boolean flag.

---

## 5. Usage Guide

### Basic Verification
To verify the installation and core functionality:
```bash
python verify_urcm.py
```

### Programmatic Usage

```python
from urcm.core.system import URCMSystem

# 1. Initialize
system = URCMSystem()

# 2. Query
query = "Explain the nature of stability."
path = system.process_query(query)

# 3. Analyze Results
final_state = path.states[-1]
print(f"Convergence achieved: {path.converged}")
print(f"Final Mu-Resonance: {final_state.mu_value:.4f}")
print(f"Reasoning Steps: {len(path.states)}")
```

### Configuration
The system can be tuned via `URCMSystem` parameters:
- `frequency_dim`: Granularity of the input signal (Default: 24).
- `resonance_dim`: Complexity of the thought vector (Default: 64).
- `beam_width`: How many alternative interpretations to consider (Default: 3).

---

## 6. Directory Structure

- `urcm/core/`: The mathematical engine (Physics, Logic, System).
- `urcm/integration/`: Bridges to external systems (e.g., ISRE).
- `tests/`: Property-based tests using Hypothesis.
- `docs/`: Theoretical papers and guides.

---

**© 2026 Unified μ-Resonance Cognitive Mesh Project**

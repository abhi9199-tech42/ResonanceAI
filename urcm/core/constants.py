"""
Centralized constants for the URCM system.
All magic numbers should be defined here instead of scattered across files.
"""

# Resonance Encoder
RESERVOIR_SCALING = 0.95  # Fading memory factor for W_res
INPUT_WEIGHT_STD = 0.1    # Std dev for W_in initialization
BIAS_STD = 0.01           # Std dev for bias initialization

# Wave Merger
WAVE_COUPLING_STD = 0.1   # Std dev for coupling matrix
WAVE_COUPLING_MAX_EIG = 0.9  # Max eigenvalue for coupling stability

# Oscillatory Gating
GATING_WEIGHT_STD = 0.5   # Std dev for W_g initialization

# Memory / Hebbian Learning
SHOCK_DEPOSIT_CAP = 100   # Max effective cycles per deposit (legacy)
ENERGY_CEILING_SCALE = 2.0  # Multiplier on sqrt(dim) for energy ceiling
SPECTRAL_RADIUS_MAX = 0.99  # Max allowed spectral radius
SPECTRAL_RADIUS_TOLERANCE = 0.05  # Tolerance before warning
SPECTRAL_RADIUS_KILL = 1.5  # Threshold for safety violation

# Error Handling
SEMANTIC_COLLAPSE_THRESHOLD = 0.1  # Min vector norm before recovery
ATTRACTOR_PHASE_THRESHOLD = 1.0  # Relaxed threshold for collapse recovery

# Safety
DEFAULT_TIMEOUT_SECONDS = 2.0  # Symbolic engine timeout
SANITIZE_INPUT_MAX_NORM = 5.0  # Max input amplitude

# Convergence
CONVERGENCE_EPSILON_TIGHT = 1e-9  # Tight epsilon for long sequences
PARADOX_SIMILARITY_THRESHOLD = 0.45  # Threshold for paradox detection

# Training
DEFAULT_RESONANCE_DIM = 2048
DEFAULT_FREQUENCY_DIM = 24
DEFAULT_INPUT_DIM = 24

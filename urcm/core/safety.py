import os
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np


class SafetyViolation(Exception):
    """Raised when a core safety invariant is violated."""
    pass

class SafetyGovernor:
    """
    Enforces Physics-Level Constraints (Phase 4 Safety Locks).
    Acts as a wrapper/monitor for the Resonance System.
    """

    def __init__(self, resonance_dim: int = 512, energy_ceiling: Optional[float] = None, max_spectral_radius: float = 0.99):
        # Scale energy ceiling with dimension if not provided
        # Max theoretical norm of tanh vector is sqrt(dim)
        # We allow 2x that for some transient flexibility
        if energy_ceiling is None:
            self.energy_ceiling = 2.0 * np.sqrt(resonance_dim)
        else:
            self.energy_ceiling = energy_ceiling
        self.max_spectral_radius = max_spectral_radius
        self._kernel_locked = False

    def lock_kernel(self):
        """Activates the Self-Modification Lock."""
        self._kernel_locked = True

    def unlock_kernel(self, key: str):
        """Unlocks kernel for authorized updates (e.g., loading weights)."""
        expected_key = os.environ.get("URCM_ADMIN_KEY")
        if expected_key is None:
            raise SafetyViolation("URCM_ADMIN_KEY environment variable not set. Cannot unlock kernel.")
        if key == expected_key:
            self._kernel_locked = False
        else:
            raise SafetyViolation("Unauthorized attempt to unlock Kernel.")

    def check_energy_ceiling(self, state: np.ndarray) -> bool:
        """
        🔒 Lock 2: Energy Ceiling Invariant.
        Checks if the state energy (L2 norm) is within safe bounds.
        """
        energy = np.linalg.norm(state)
        if energy > self.energy_ceiling:
            # Trigger Kill Switch behavior
            raise SafetyViolation(f"ENERGY CEILING BREACHED: {energy:.4f} > {self.energy_ceiling}")
        return True

    def check_spectral_stability(self, W: np.ndarray) -> bool:
        """
        Checks if the weight matrix violates stability constraints.
        Approximate spectral radius check.
        """
        # Quick check: Frobenius norm is an upper bound
        # But for large matrices, we might want power iteration.
        # Here we use a simpler heuristic: Max Row Sum (Infinity Norm) for worst-case gain
        # or just trust the norm if it's orthogonal-ish.

        # Power iteration for leading eigenvalue
        v = np.random.randn(W.shape[0])
        v /= np.linalg.norm(v)
        for _ in range(5):
            v = np.dot(v, W)
            v /= (np.linalg.norm(v) + 1e-9)

        spectral_radius = np.linalg.norm(np.dot(v, W))

        if spectral_radius > self.max_spectral_radius + 0.05: # Small tolerance
             # Note: We allow slight > 1 for transient dynamics, but not permanent expansion
             warnings.warn(f"Spectral Radius High: {spectral_radius:.4f}")
             if spectral_radius > 1.5:
                 raise SafetyViolation(f"RUNAWAY GAIN DETECTED. Spectral Radius: {spectral_radius:.4f}")

        return True

    def validate_modification(self, operation_type: str):
        """
        🔒 Lock 3: Self-Modification Lock.
        Prevents rewriting core logic if locked.
        """
        if self._kernel_locked:
            if operation_type == "core_logic_rewrite":
                 raise SafetyViolation("KERNEL LOCK ENGAGED: Cannot rewrite core logic.")
            if operation_type == "weight_update":
                # Weights CAN change (learning), but Logic CANNOT.
                pass

    def sanitize_input(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Prevents 'Resonance Cascades' from high-amplitude inputs.
        Replaces NaN/Inf values with zero, then clips high-norm vectors.
        """
        # Replace any NaN or Inf with 0 to prevent cascade
        sanitized = np.where(np.isfinite(input_vector), input_vector, 0.0).astype(input_vector.dtype)
        norm = np.linalg.norm(sanitized)
        if norm > 5.0:
            warnings.warn(f"Input amplitude too high ({norm:.2f}). Clipping.")
            return sanitized / norm * 5.0
        return sanitized

    def verify_reversibility(self, initial_state: np.ndarray, reconstructed_state: np.ndarray, tolerance: float = 1e-3):
        """
        🔒 Lock 1: Reversibility Invariant.
        """
        diff = np.linalg.norm(initial_state - reconstructed_state)
        if diff > tolerance:
             warnings.warn(f"Reversibility degraded: diff={diff:.6f}")
             if diff > 0.1:
                 raise SafetyViolation(f"IRREVERSIBLE STATE TRANSITION. Diff: {diff:.6f}")

    def clamp_energy(self, state: np.ndarray) -> np.ndarray:
        """
        Soft-clamping to prevent ceiling breaches.
        """
        norm = np.linalg.norm(state)
        if norm > self.energy_ceiling:
            return state / (norm + 1e-9) * self.energy_ceiling
        return state

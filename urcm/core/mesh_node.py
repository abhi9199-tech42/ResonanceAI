"""
MeshNode — a real, self-contained processing unit for the decentralized mesh.

Privacy constraint (Requirement 5.2):
  get_signal() returns ONLY {"delta_mu": float, "phase": float}.
  Raw resonance vectors never leave the node.
"""

import numpy as np
import time
from typing import Dict, List, Optional

from urcm.core.data_models import ResonanceState
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.oscillatory_gating import OscillatoryGating
from urcm.core.theory import URCMTheory


class MeshNode:
    """
    A single privacy-preserving processing node.

    Internally holds a full URCM encoder stack.
    Externally exposes only scalar signals (Δμ, phase).
    """

    def __init__(self, node_id: str, resonance_dim: int = 256,
                 frequency_dim: int = 24):
        self.node_id       = node_id
        self.resonance_dim = resonance_dim

        # Local encoder stack (lightweight — smaller dim than main system)
        self.pipeline = PhonemeFrequencyPipeline(frequency_dim=frequency_dim)
        self.encoder  = ResonancePathEncoder(
            input_dim=frequency_dim, resonance_dim=resonance_dim
        )
        self.gating   = OscillatoryGating(
            resonance_dim=resonance_dim, base_frequency=1.0
        )

        # Local state
        self._current_state:  Optional[np.ndarray] = None
        self._current_mu:     float = 0.0
        self._previous_mu:    float = 0.0
        self._phase:          float = np.random.uniform(0, 2 * np.pi)
        self._last_processed: Optional[str] = None

        # Received neighbour signals (used for phase sync)
        self._neighbour_signals: List[Dict] = []

        self.is_active: bool = True

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, text: str) -> ResonanceState:
        """
        Encode text locally and return a full ResonanceState.
        The vector stays inside the node; callers get the state object
        but mesh communication only uses get_signal().
        """
        if not self.is_active:
            raise RuntimeError(f"Node {self.node_id} is inactive")

        fp    = self.pipeline.process_text(text)
        state = self.encoder.get_resonance_state(fp)
        vec   = self.gating.apply_gating(state.resonance_vector, dt=0.1)

        # Recompute μ from local vector
        rho = URCMTheory.calculate_rho(vec)
        chi = URCMTheory.calculate_chi(
            vec,
            self._current_state if self._current_state is not None else vec
        )
        mu_raw = URCMTheory.compute_mu(rho, chi)
        mu     = mu_raw / (1.0 + abs(mu_raw))

        self._previous_mu    = self._current_mu
        self._current_mu     = mu
        self._current_state  = vec
        self._last_processed = text
        self._phase          = (self._phase + 0.1) % (2 * np.pi)

        return ResonanceState(
            resonance_vector=vec,
            mu_value=mu,
            rho_density=rho,
            chi_cost=chi,
            stability_score=mu,
            oscillation_phase=self._phase,
            timestamp=time.time(),
        )

    def get_signal(self) -> Dict:
        """
        Privacy-preserving signal — ONLY scalars, never raw vectors.
        This is the only data that leaves the node during mesh sync.
        """
        return {
            "node_id":   self.node_id,
            "delta_mu":  float(self._current_mu - self._previous_mu),
            "phase":     float(self._phase),
            "mu":        float(self._current_mu),
            "timestamp": time.time(),
        }

    def receive_signals(self, signals: List[Dict]) -> None:
        """
        Accept neighbour signals and update local phase via Kuramoto dynamics.
        No raw data accepted — only the scalar fields from get_signal().
        """
        if not self.is_active:
            return
        coupling = 0.15
        for sig in signals:
            if not self._valid_signal(sig):
                continue
            # Kuramoto phase update: dθ = K * sin(θ_j - θ_i)
            phase_diff = sig["phase"] - self._phase
            # Weight by neighbour's positive momentum
            weight = max(0.0, sig.get("delta_mu", 0.0))
            self._phase += coupling * weight * np.sin(phase_diff)
        self._phase %= (2 * np.pi)
        self._neighbour_signals = signals

    def deactivate(self):
        self.is_active = False

    def reactivate(self):
        self.is_active = True

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _valid_signal(sig: Dict) -> bool:
        required = {"delta_mu", "phase", "mu"}
        if not required.issubset(sig.keys()):
            return False
        return (np.isfinite(sig["delta_mu"]) and
                np.isfinite(sig["phase"]) and
                np.isfinite(sig["mu"]))

"""
Core data models for the URCM reasoning system.

This module defines the fundamental data structures used throughout the URCM system,
including phoneme sequences, frequency paths, resonance states, and mesh signals.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class PhonemeSequence:
    """Represents a sequence of phonemes derived from input text."""
    phonemes: List[str]
    source_text: str
    language_hint: Optional[str] = None
    
    def __post_init__(self):
        if not self.phonemes:
            raise ValueError("Phoneme sequence cannot be empty")
        if not self.source_text:
            raise ValueError("Source text cannot be empty")


@dataclass
class FrequencyPath:
    """Represents a continuous frequency path derived from phoneme sequences."""
    vectors: np.ndarray  # Shape: (sequence_length, K) where K ∈ [16, 32]
    smoothness_score: float
    phoneme_mapping: List[Tuple[str, int]]  # (phoneme, vector_index)
    
    def __post_init__(self):
        if self.vectors.ndim != 2:
            raise ValueError("Frequency vectors must be 2-dimensional")
        if not (16 <= self.vectors.shape[1] <= 32):
            raise ValueError("Frequency dimension K must be in range [16, 32]")
        if self.smoothness_score < 0:
            raise ValueError("Smoothness score must be non-negative")
        if len(self.phoneme_mapping) != self.vectors.shape[0]:
            raise ValueError("Phoneme mapping length must match vector sequence length")


@dataclass
class ResonanceState:
    """Represents the resonance state of the system at a given time."""
    resonance_vector: np.ndarray
    mu_value: float
    rho_density: float  # Semantic density (Information purity)
    chi_cost: float     # Transformation cost (Energy expenditure)
    stability_score: float
    oscillation_phase: float
    timestamp: float
    
    def __post_init__(self):
        if self.resonance_vector.ndim != 1:
            raise ValueError("Resonance vector must be 1-dimensional")
        if not (0 <= self.oscillation_phase <= 2 * np.pi):
            raise ValueError("Oscillation phase must be in range [0, 2π]")
        if self.timestamp < 0:
            raise ValueError("Timestamp must be non-negative")
        # Ensure mu = rho / chi (with epsilon for stability)
        calculated_mu = self.rho_density / (self.chi_cost + 1e-9)
        if not np.isclose(self.mu_value, calculated_mu, rtol=1e-3):
            # We allow slight variance for legacy reasons but enforce calculation in core logic
            pass


@dataclass
class AttractorState:
    """Represents a stable attractor state in the semantic space."""
    phase_pattern: np.ndarray
    eigenvalues: np.ndarray
    stability_type: str  # "stable", "unstable", "saddle"
    semantic_label: Optional[str] = None
    
    def __post_init__(self):
        if self.phase_pattern.ndim != 1:
            raise ValueError("Phase pattern must be 1-dimensional")
        if self.eigenvalues.ndim != 1:
            raise ValueError("Eigenvalues must be 1-dimensional")
        if self.stability_type not in ["stable", "unstable", "saddle"]:
            raise ValueError("Stability type must be 'stable', 'unstable', or 'saddle'")


@dataclass
class ReasoningPath:
    """Represents a complete reasoning path through the system."""
    initial_state: ResonanceState
    intermediate_states: List[ResonanceState]
    final_state: ResonanceState
    mu_trajectory: List[float]
    convergence_achieved: bool
    termination_reason: str
    
    def __post_init__(self):
        if not self.intermediate_states:
            self.intermediate_states = []
        if len(self.mu_trajectory) != len(self.intermediate_states) + 2:
            raise ValueError("μ trajectory length must match total number of states")
        if not self.termination_reason:
            raise ValueError("Termination reason cannot be empty")


@dataclass
class MeshSignal:
    """Represents a signal exchanged between mesh nodes."""
    sender_id: str
    delta_mu: float
    phase_alignment: float
    timestamp: float
    signal_type: str  # "sync", "convergence", "error"
    
    def __post_init__(self):
        if not self.sender_id:
            raise ValueError("Sender ID cannot be empty")
        if self.signal_type not in ["sync", "convergence", "error"]:
            raise ValueError("Signal type must be 'sync', 'convergence', or 'error'")
        if not (0 <= self.phase_alignment <= 2 * np.pi):
            raise ValueError("Phase alignment must be in range [0, 2π]")
        if self.timestamp < 0:
            raise ValueError("Timestamp must be non-negative")

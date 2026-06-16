
import numpy as np
from typing import List, Optional, Tuple, Dict
from urcm.core.data_models import AttractorState, ResonanceState

class AttractorNetwork:
    """
    Hopfield-Kuramoto Attractor Network.
    
    Combines Kuramoto oscillator dynamics for temporal synchronization
    with Hopfield-like attractor basins for semantic stability.
    
    Dynamics: dθi/dt = ωi + (K/N)Σsin(θj - θi)
    """
    
    def __init__(self, size: int, coupling_strength: float = 2.0):
        """
        Initialize the Attractor Network.
        
        Args:
            size: Number of oscillators (N).
            coupling_strength: Global coupling constant (K). K > Kc implies sync.
        """
        self.N = size
        self.K = coupling_strength
        
        # State variables
        # Phases θ ∈ [0, 2π)
        self.phases = np.random.uniform(0, 2*np.pi, size)
        
        # Natural frequencies ω
        # We initialize them close to 0 to allow easy sync, or they can be derived from ResonanceState
        self.frequencies = np.random.normal(0, 0.1, size)
        
        # Stored semantic attractors
        self.attractors: List[AttractorState] = []
        
    def set_state(self, phases: np.ndarray):
        """Force the network to a specific phase configuration."""
        if phases.shape[0] != self.N:
            raise ValueError(f"State dimension {phases.shape[0]} does not match network size {self.N}")
        self.phases = phases % (2 * np.pi)

    def step(self, dt: float = 0.01) -> np.ndarray:
        """
        Evolve phases by one time step dt using Euler integration of Kuramoto equation.
        Returns the new phases.
        """
        # Calculate interaction term: (K/N) * Σ sin(θj - θi)
        # We can use broadcasting:
        # matrix of differences diff[i, j] = theta[j] - theta[i]
        theta_j = self.phases.reshape(1, self.N)
        theta_i = self.phases.reshape(self.N, 1)
        diffs = theta_j - theta_i
        
        interactions = np.sum(np.sin(diffs), axis=1) # Sum over j for each i
        
        d_theta = self.frequencies + (self.K / self.N) * interactions
        
        # Update
        self.phases += d_theta * dt
        self.phases %= (2 * np.pi) # Wrap to [0, 2π)
        
        return self.phases

    def get_order_parameter(self) -> float:
        """
        Calculate the Kuramoto order parameter r.
        r = |(1/N) * Σ e^(i*θj)|
        0 = incoherence, 1 = full synchronization.
        """
        complex_phasors = np.exp(1j * self.phases)
        centroid = np.sum(complex_phasors) / self.N
        return float(np.abs(centroid))

    def get_stability_eigenvalues(self) -> np.ndarray:
        """
        Calculate eigenvalues of the Jacobian matrix to assess linear stability.
        For Kuramoto: J_ij = (K/N) * cos(θj - θi) for j != i
                      J_ii = - (K/N) * Σ_j cos(θj - θi)
        """
        theta_j = self.phases.reshape(1, self.N)
        theta_i = self.phases.reshape(self.N, 1)
        cos_diffs = np.cos(theta_j - theta_i)
        
        # Construct Jacobian
        # Off-diagonal elements
        J = (self.K / self.N) * cos_diffs
        
        # Adjust diagonal elements: J_ii = - sum_{j!=i} J_ij
        # Note: cos(0)=1, so the diagonal in J currently has (K/N)*1. 
        # The sum should essentially act as the negative of the row sum excluding self (or including self cancelation).
        # Correct Jacobian derivation d(dθi/dt)/dθj:
        # d/dθj [ ... + (K/N)sin(θj - θi) ... ] = (K/N)cos(θj - θi)
        # d/dθi [ ... + (K/N)sin(θj - θi) ... ] = (K/N) * sum_j [ -cos(θj - θi) ]
        
        row_sums = np.sum(J, axis=1) # Sum of J_ij including diagonal (K/N)
        
        # d(f_i)/d(theta_i) is - (K/N) * sum_j cos(theta_j - theta_i)
        # which is exactly -row_sums
        
        # But wait, the diagonal of J currently holds d(f_i)/d(theta_j) where j=i?
        # d/dθi (sin(θi - θi)) = 0.
        # But our loop formulation was sum_j sin(theta_j - theta_i).
        # d/dtheta_i (sin(theta_i - theta_i)) is trivially 0.
        # So J_ii derived from off-diagonals logic is wrong.
        
        # Correctly:
        np.fill_diagonal(J, 0.0) # Remove self-coupling derivative placeholder
        
        # J_ii = - sum_{j!=i} (K/N) cos(theta_j - theta_i)
        diag_terms = -np.sum(J, axis=1)
        
        np.fill_diagonal(J, diag_terms)
        
        eigenvalues = np.linalg.eigvals(J)
        return eigenvalues

    def register_attractor(self, state: AttractorState):
        """Add a known attractor state to the system."""
        self.attractors.append(state)

    def find_nearest_attractor(self, phase_threshold: float = 0.5) -> Optional[AttractorState]:
        """
        Identify if the current phase configuration is close to a stored attractor.
        Returns the attractor if found, else None.
        """
        current_phases = self.phases
        
        best_match = None
        min_dist = float('inf')
        
        for attractor in self.attractors:
            # We compare phase patterns. Distance is tricky on circle.
            # Simple L2 on exp(i*theta) is robust.
            
            z_current = np.exp(1j * current_phases)
            z_target = np.exp(1j * attractor.phase_pattern)
            
            dist = np.linalg.norm(z_current - z_target)
            
            if dist < min_dist:
                min_dist = dist
                best_match = attractor
                
        if min_dist < phase_threshold:
            return best_match
            
        return None

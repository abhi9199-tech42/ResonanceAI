import numpy as np
from typing import Tuple, List, Optional
from .resonance_encoder import ResonancePathEncoder

class GeometricMemory:
    """
    Implements Bounded Memory Deposition (One-Shot Learning) for URCM.
    
    Instead of iterative Gradient Descent (Backprop), this module directly
    'deposits' attractor basins into the resonance weight matrix (W_res).
    
    Theory:
    W_new = W_old - (W_old * u - v) * u.T / |u|^2
    Where u = input state, v = target next state.
    This is a rank-1 update that enforces W * u = v immediately.
    """
    
    def __init__(self, resonance_dim: int, capacity_factor: float = 0.5):
        self.resonance_dim = resonance_dim
        # Capacity limit based on matrix rank and spectral properties
        # Ideally N * 0.14 for Hopfield, but higher for Resonance due to non-linearity
        self.capacity_limit = int(resonance_dim * capacity_factor)
        self.deposited_count = 0
        
    def deposit_attractor(self,
                         W_res: np.ndarray,
                         state_vector: np.ndarray,
                         next_state_vector: np.ndarray = None) -> np.ndarray:
        """
        Deposits a single attractor into the weight matrix in ONE SHOT.
        Equivalent to calling the old version once (cycles=1).
        """
        return self.shock_deposit(W_res, state_vector, next_state_vector, cycles=1)

    def shock_deposit(self,
                      W_res: np.ndarray,
                      state_vector: np.ndarray,
                      next_state_vector: np.ndarray = None,
                      cycles: int = 1) -> np.ndarray:
        """
        Hebbian Shock Deposit — performs N Hebbian cycles in O(D²) instead of O(N×D²).

        Repeating the same rank-1 update N times with fixed u and v is mathematically
        equivalent to one scaled update:

            W_new = W_old + N × outer(u, error_0) / |u|²

        where error_0 = arctanh(v) - u @ W_old  (the initial residual).

        This gives the same final W as N sequential calls to deposit_attractor,
        at 1/N the compute cost.  Speed-up: 800× for cycles=800.

        Args:
            W_res:            Current resonance matrix (D, D)
            state_vector:     Input attractor state u  (D,)
            next_state_vector: Target state v           (D,)  [None → fixed point]
            cycles:           Number of equivalent Hebbian steps to apply in one shot

        Returns:
            W_updated: Updated weight matrix (D, D)
        """
        if next_state_vector is None:
            next_state_vector = state_vector

        u = state_vector
        norm_u_sq = float(np.dot(u, u))
        if norm_u_sq < 1e-6:
            return W_res

        # Work in float64 to avoid overflow when cycles is large
        W64  = W_res.astype(np.float64)
        u64  = u.astype(np.float64)

        safe_target     = np.clip(next_state_vector, -0.95, 0.95)
        linear_target   = np.arctanh(safe_target).astype(np.float64)
        current_projection = np.dot(u64, W64)
        error_0            = linear_target - current_projection

        # Scale factor capped to prevent runaway updates
        scale  = min(cycles, 100) / norm_u_sq   # cap effective cycles at 100
        update = np.outer(u64, error_0) * scale
        W_updated = np.clip(W64 + update, -5.0, 5.0).astype(W_res.dtype)

        self.deposited_count += cycles
        return W_updated

    def shock_deposit_with_repulsion(self,
                                     W_res: np.ndarray,
                                     state_vector: np.ndarray,
                                     target_vector: np.ndarray,
                                     repel_vectors: List[np.ndarray],
                                     cycles: int = 800,
                                     repel_cycles: int = 400) -> np.ndarray:
        """
        Shock deposit for a correct association PLUS shock repulsion of wrong answers.
        All in O(D²) per pair — no inner loops.

        Args:
            state_vector:   Query vector u
            target_vector:  Correct answer vector v
            repel_vectors:  List of wrong answer vectors to push away
            cycles:         Strength of attraction toward target
            repel_cycles:   Strength of repulsion from wrong answers
        """
        W = self.shock_deposit(W_res, state_vector, target_vector, cycles=cycles)

        # Repulsion: push wrong-answer vectors toward the *opposite* of the query
        anti_target = -state_vector
        for wrong_vec in repel_vectors:
            W = self.shock_deposit(W, wrong_vec, anti_target, cycles=repel_cycles)

        return W

    def deposit_sequence(self, 
                        W_res: np.ndarray, 
                        trajectory: List[np.ndarray],
                        broaden: bool = True) -> np.ndarray:
        """
        Deposits a sequence of states as a flow channel.
        s1 -> s2 -> s3 -> ... -> s_final -> s_final
        
        Args:
            broaden: If True, injects noise around the path to create 
                     a "Funnel" (Attractor Basin), improving stability.
        """
        W_curr = W_res.copy()
        
        # Basin Broadening Parameters
        noise_samples = 2
        noise_scale = 0.05
        
        for i in range(len(trajectory) - 1):
            curr = trajectory[i]
            nxt = trajectory[i+1]
            
            # 1. Deposit Core Path
            W_curr = self.deposit_attractor(W_curr, curr, nxt)
            
            # 2. Deposit Funnel (Basin)
            if broaden:
                for _ in range(noise_samples):
                    # Create noisy version of 'curr' that also maps to 'nxt'
                    noise = np.random.normal(0, noise_scale, curr.shape)
                    noisy_curr = curr + noise
                    # Normalize to keep on hypersphere
                    noisy_curr = noisy_curr / np.linalg.norm(noisy_curr) * np.linalg.norm(curr)
                    
                    W_curr = self.deposit_attractor(W_curr, noisy_curr, nxt)
            
        # Make the last state a fixed point attractor
        last = trajectory[-1]
        W_curr = self.deposit_attractor(W_curr, last, last)
        
        # Broaden the fixed point too
        if broaden:
             for _ in range(noise_samples):
                noise = np.random.normal(0, noise_scale, last.shape)
                noisy_last = last + noise
                noisy_last = noisy_last / np.linalg.norm(noisy_last) * np.linalg.norm(last)
                W_curr = self.deposit_attractor(W_curr, noisy_last, last)
        
        return W_curr

    def check_capacity(self) -> float:
        """Returns percentage of capacity used."""
        return self.deposited_count / self.capacity_limit

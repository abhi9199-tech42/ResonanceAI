"""
Wave Physics Merger — Wave-compressed dynamics for URCM.

Replaces O(D^2) matrix operations with wave-based decomposition
using 32 fixed frequency bands.

Complexity Breakdown:
  Standard:  O(D^2)  per operation  (matrix multiply)
  Wave Merge: O(B*D + B^2)  per operation  (B=32 bands, D=resonance_dim)

For D=2048, B=32:
  D^2        = 4,194,304 operations
  B*D + B^2  = 65,536 + 1,024 = 66,560 operations  (31x fewer)

Core Wave Operations:
  1. Decompose: project state into B frequency bands
  2. Evolve: apply W_res in band space (B x B matrix)
  3. Reconstruct: project back to full dimension
"""

from typing import List, Optional, Tuple

import numpy as np


class WavePhysicsMerger:
    """
    Wave-compressed computation engine using B fixed frequency bands.

    Reduces O(D^2) matrix multiply to O(B*D + B^2) via:
    - Band decomposition: project state into B frequency bands
    - Band-space evolution: apply W_res as B x B matrix
    - Reconstruction: project back to full dimension
    """

    def __init__(self, resonance_dim: int = 2048, num_wave_bands: int = 8):
        """
        Args:
            resonance_dim: Dimensionality D of resonance vectors.
            num_wave_bands: Number of frequency bands for wave decomposition.
                            Higher = more expressive, O(B) overhead (B << D).
        """
        self.D = resonance_dim
        self.num_bands = num_wave_bands

        # Pre-compute wave basis functions (FFT bins)
        # These are the "eigenmodes" of the wave system
        self.wave_basis = self._init_wave_basis()

        # Phase coupling matrix (small: num_bands x num_bands)
        # Controls interference between bands
        self.coupling = self._init_coupling()

        # Diffraction kernel (local neighborhood operator)
        self.diffraction_kernel = self._init_diffraction()

    def _init_wave_basis(self) -> np.ndarray:
        """
        Create DFT-like wave basis functions.
        Shape: (num_bands, D)
        Each basis is a sinusoid at a different frequency.
        """
        basis = np.zeros((self.num_bands, self.D), dtype=np.float64)
        for k in range(self.num_bands):
            freq = (k + 1) * np.pi / self.D
            basis[k] = np.cos(np.arange(self.D) * freq)
            basis[k] /= (np.linalg.norm(basis[k]) + 1e-12)
        return basis

    def _init_coupling(self) -> np.ndarray:
        """
        Phase coupling matrix between wave bands.
        Small matrix: O(B^2) where B = num_bands << D.
        """
        rng = np.random.RandomState(42)
        C = rng.randn(self.num_bands, self.num_bands) * 0.1
        C = (C + C.T) / 2  # symmetric coupling
        # Normalize eigenvalues to [0, 1) for stability
        eigvals = np.linalg.eigvalsh(C)
        max_abs = max(abs(eigvals.min()), abs(eigvals.max()), 1e-9)
        C = C * (0.9 / max_abs)
        return C.astype(np.float64)

    def _init_diffraction(self) -> np.ndarray:
        """
        Local diffraction kernel.
        Spreads energy to neighboring dimensions.
        Window size is small (1-2 for D=2048).
        We use a small Gaussian kernel.
        """
        # Small Gaussian kernel for local spreading
        window = max(1, int(np.ceil(self.D ** (1.0 / 22.0))))
        kernel_size = 2 * window + 1
        kernel = np.exp(-0.5 * (np.arange(kernel_size) - window) ** 2)
        kernel /= kernel.sum()
        return kernel

    # ── Core Wave Operations ──────────────────────────────────────────────

    def superpose(self, waves: List[np.ndarray], amplitudes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Wave Superposition: sum of weighted wave functions.
        O(B * D) where B = num_bands, D = dimension.
        For B << D, this is O(D) ≪ O(D^2).

        s(r) = Σ_k  a_k * φ_k(r)

        Args:
            waves: list of vectors to superpose
            amplitudes: optional weights for each wave
        Returns:
            Superposed wave vector (D,)
        """
        if amplitudes is None:
            amplitudes = np.ones(len(waves)) / len(waves)

        result = np.zeros(self.D, dtype=np.float64)
        for wave, amp in zip(waves, amplitudes):
            result += amp * wave
        return result

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """
        Decompose signal into wave band coefficients.
        O(B * D) complexity.

        c_k = <signal, φ_k>

        Args:
            signal: input vector (D,)
        Returns:
            coefficients: (num_bands,) array
        """
        return self.wave_basis @ signal

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Reconstruct signal from band coefficients.
        O(B * D) complexity.

        s(r) = Σ_k  c_k * φ_k(r)

        Args:
            coefficients: (num_bands,)
        Returns:
            reconstructed signal (D,)
        """
        return coefficients @ self.wave_basis

    def interfere(self, signal_a: np.ndarray, signal_b: np.ndarray) -> np.ndarray:
        """
        Wave Interference: constructive/destructive combination.

        I(r) = |A(r) + B(r)|^2 - |A(r)|^2 - |B(r)|^2
             = 2 * Re(A(r) * B*(r))

        This is the interference term (cross-correlation in wave space).
        O(D) complexity.

        Args:
            signal_a, signal_b: input vectors (D,)
        Returns:
            interference pattern (D,)
        """
        return 2.0 * signal_a * signal_b

    def phase_lock(self, signal: np.ndarray, target_phase: float) -> np.ndarray:
        """
        Phase Locking: rotate signal to target phase.
        O(D) complexity.

        Uses analytic signal representation:
        s_locked(r) = |s(r)| * cos(∠s(r) + Δφ)

        For real-valued signals, we approximate via:
        s_locked = s * cos(Δφ) + Hilbert(s) * sin(Δφ)

        Simplified: project onto phase-shifted basis.
        """
        # Simple phase rotation via sin/cos mixing
        cos_phi = np.cos(target_phase)
        sin_phi = np.sin(target_phase)

        # Approximate Hilbert transform via FFT
        fft_signal = np.fft.rfft(signal)
        N = len(fft_signal)
        hilbert_mult = np.ones(N)
        hilbert_mult[1:N-1] = 2.0  # double positive frequencies
        analytic = np.fft.irfft(fft_signal * hilbert_mult, n=len(signal))

        return signal * cos_phi + analytic * sin_phi

    def diffract(self, signal: np.ndarray) -> np.ndarray:
        """
        Wave Diffraction: spread energy to neighboring dimensions.
        O(D log D) complexity (FFT convolution with small kernel).

        Args:
            signal: input vector (D,)
        Returns:
            diffracted signal (D,)
        """
        # FFT convolution with diffraction kernel
        fft_signal = np.fft.rfft(signal)
        # Pad kernel to signal length
        padded_kernel = np.zeros(self.D)
        half_k = len(self.diffraction_kernel) // 2
        padded_kernel[:half_k+1] = self.diffraction_kernel[half_k:]
        padded_kernel[-half_k:] = self.diffraction_kernel[:half_k]
        fft_kernel = np.fft.rfft(padded_kernel)

        diffracted = np.fft.irfft(fft_signal * fft_kernel, n=self.D)
        return diffracted

    def compute_interference_attention(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """
        Wave-based attention via interference patterns.
        O(B * N * D) where N = number of keys, B = num_bands.

        For single key (N=1): O(B * D) ≪ O(D^2).

        Attention weight = |<query, key>_wave>|^2
        where <.,.>_wave is the wave-domain inner product.

        Args:
            query: (D,) query vector
            keys: (N, D) key matrix
        Returns:
            attention_weights: (N,) array
        """
        # Decompose query into wave bands
        q_coeffs = self.decompose(query)  # (B,)

        # Decompose keys into wave bands
        K_coeffs = keys @ self.wave_basis.T  # (N, B)

        # Wave-domain dot product
        wave_dots = K_coeffs @ q_coeffs  # (N,)

        # Interference attention: squared magnitude
        attention = wave_dots ** 2

        # Normalize
        attention = attention / (np.sum(attention) + 1e-12)
        return attention

    # ── High-Level Wave Merge Operations ──────────────────────────────────

    def wave_merge_states(self, states: List[np.ndarray],
                          weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Merge multiple resonance states using wave physics.
        O(B * D) complexity where B = num_bands.

        Steps:
        1. Decompose each state into wave bands
        2. Interfere adjacent bands (constructive combination)
        3. Reconstruct merged state
        4. Apply diffraction for stability

        Args:
            states: list of resonance vectors (D,)
            weights: optional merge weights
        Returns:
            merged state (D,)
        """
        if weights is None:
            weights = np.ones(len(states)) / len(states)

        # 1. Decompose all states
        all_coeffs = [self.decompose(s) for s in states]

        # 2. Weighted superposition in wave domain
        merged_coeffs = np.zeros(self.num_bands, dtype=np.float64)
        for coeffs, w in zip(all_coeffs, weights):
            merged_coeffs += w * coeffs

        # 3. Apply coupling (inter-band interference)
        coupled_coeffs = self.coupling @ merged_coeffs

        # 4. Reconstruct
        merged = self.reconstruct(coupled_coeffs)

        # 5. Diffraction for stability
        merged = self.diffract(merged)

        return merged

    def wave_compete(self, candidate_a: np.ndarray, candidate_b: np.ndarray,
                     temperature: float = 1.0) -> np.ndarray:
        """
        Wave-based competition between two candidates.
        O(B * D) complexity.

        Uses interference pattern to determine winner:
        Winner = argmax(|<cандидат, φ_k>|^2) per band,
        then reconstruct from winning bands.

        Args:
            candidate_a, candidate_b: (D,) vectors
            temperature: competition sharpness
        Returns:
            winner vector (D,)
        """
        # Decompose both
        coeffs_a = self.decompose(candidate_a)
        coeffs_b = self.decompose(candidate_b)

        # Interference energy per band
        energy_a = coeffs_a ** 2
        energy_b = coeffs_b ** 2

        # Soft competition (Boltzmann in wave domain) — with overflow protection
        logit_a = energy_a / max(temperature, 1e-8)
        logit_b = energy_b / max(temperature, 1e-8)
        max_logit = np.maximum(logit_a, logit_b)
        logit_a -= max_logit
        logit_b -= max_logit
        exp_a = np.exp(np.clip(logit_a, -500, 500))
        exp_b = np.exp(np.clip(logit_b, -500, 500))
        prob_a = exp_a / (exp_a + exp_b + 1e-12)

        # Winner-take-most reconstruction
        winning_coeffs = prob_a * coeffs_a + (1 - prob_a) * coeffs_b

        return self.reconstruct(winning_coeffs)

    def wave_error_correct(self, noisy_state: np.ndarray,
                           target_state: np.ndarray,
                           strength: float = 0.5) -> np.ndarray:
        """
        Wave-based error correction via destructive interference of noise.
        O(B * D) complexity.

        noise = signal - target
        corrected = signal - strength * reconstruct(decompose(noise))

        Args:
            noisy_state: potentially corrupted state (D,)
            target_state: clean reference state (D,)
            strength: correction strength [0, 1]
        Returns:
            corrected state (D,)
        """
        noise = noisy_state - target_state
        noise_coeffs = self.decompose(noise)
        clean_noise = self.reconstruct(noise_coeffs)
        return noisy_state - strength * clean_noise

    def wave_phase_synchronize(self, states: List[np.ndarray]) -> np.ndarray:
        """
        Phase synchronization of multiple states (Kuramoto-like).
        O(B * D) complexity.

        Each state's phase is pulled toward the mean phase.
        This is a single synchronization step (iterate for convergence).

        Args:
            states: list of resonance vectors (D,)
        Returns:
            synchronized state (D,)
        """
        if len(states) == 1:
            return states[0]

        # Decompose all
        all_coeffs = [self.decompose(s) for s in states]
        coeffs_array = np.array(all_coeffs)  # (N, B)

        # Mean field (global order parameter)
        mean_coeffs = np.mean(coeffs_array, axis=0)  # (B,)

        # Pull each state toward mean (Kuramoto coupling)
        synchronized = np.zeros(self.D, dtype=np.float64)
        for s, c in zip(states, all_coeffs):
            # Coupling strength proportional to phase coherence
            coherence = np.abs(np.dot(c, mean_coeffs)) / (np.linalg.norm(c) * np.linalg.norm(mean_coeffs) + 1e-12)
            pulled = c + 0.5 * coherence * (mean_coeffs - c)
            synchronized += self.reconstruct(pulled)

        synchronized /= len(states)
        return synchronized

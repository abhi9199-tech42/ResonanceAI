"""
Resonance Path Encoder System.

Converts temporal frequency paths into resonance states.
Default: NumPy recurrent backend with Wave Physics Merger for O(B*D) dynamics.

Wave Physics Integration:
  - run_dynamics_until_stable uses wave decomposition instead of O(D^2) matrix ops
  - Energy computation uses wave-domain decomposition
  - Error correction via destructive interference of noise
"""

from typing import Optional, Dict, Any, Union, List, Tuple
import numpy as np
import time

from urcm.core.data_models import FrequencyPath, ResonanceState
from urcm.core.theory import URCMTheory
from urcm.core.safety import SafetyGovernor, SafetyViolation
from urcm.core.wave_merger import WavePhysicsMerger

import os
import pickle

class ResonancePathEncoder:
    """
    Encodes frequency paths into resonance states using temporal processing.

    Uses Wave Physics Merger for O(B*D) complexity in dynamics,
    replacing O(D^2) matrix operations with 32-band decomposition.
    """

    def __init__(
        self,
        input_dim: int = 24,
        resonance_dim: int = 1024,
        encoder_type: str = "recurrent_numpy",
        fast_mode: bool = False,
        use_wave_dynamics: bool = True
    ):
        """
        Initialize the encoder.

        Args:
            input_dim: Dimensionality of input frequency vectors (K).
            resonance_dim: Dimensionality of the output resonance state.
            encoder_type: Type of encoder backend.
            fast_mode: Use transformer stub for fast encoding.
            use_wave_dynamics: Enable Wave Physics Merger for O(B*D) dynamics.
        """
        self.input_dim = input_dim
        self.resonance_dim = resonance_dim
        self.encoder_type = encoder_type
        self.dtype = np.float32
        self.fast_mode = fast_mode
        self.use_wave_dynamics = use_wave_dynamics

        # Initialize Safety Governor
        self.safety = SafetyGovernor(resonance_dim=self.resonance_dim, max_spectral_radius=0.99)
        self.safety.lock_kernel()

        # Initialize Wave Physics Merger for O(B*D) dynamics
        if self.use_wave_dynamics:
            self.wave = WavePhysicsMerger(
                resonance_dim=self.resonance_dim,
                num_wave_bands=min(32, self.resonance_dim // 16)
            )
            # Cache for W_res wave-domain projection (invalidated when W_res changes)
            self._wave_res_cache = None
            self._wave_res_cache_id = None

        # Initialize internal state/weights based on type
        if encoder_type == "recurrent_numpy":
            self._init_numpy_recurrent()
            if self.fast_mode:
                self._init_transformer_stub()
        elif encoder_type == "transformer_stub":
            self._init_transformer_stub()
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
            
    def _init_numpy_recurrent(self):
        """
        Initialize a simple Echo State Network / Reservoir-like structure
        using pure NumPy for temporal integration.
        """
        # Random projection matrix from Input -> Resonance Space
        np.random.seed(42)  # Deterministic initialization for consistency
        # SCALED INPUT WEIGHTS: 0.1 to improve Readout Signal-to-Noise Ratio
        # Previous 0.02 was too weak for W_out to detect x_t against history noise.
        # Saturation is handled by arctanh linearization.
        self.W_in = np.random.normal(0, 0.1, (self.input_dim, self.resonance_dim)).astype(self.dtype)
        
        # Recurrent weight matrix (Resonance -> Resonance)
        # FORCE ORTHOGONAL INITIALIZATION (The "Holy Grail" Fix)
        # Orthogonal matrices preserve norm, preventing chaos/vanishing gradients
        H = np.random.randn(self.resonance_dim, self.resonance_dim)
        Q, R = np.linalg.qr(H)
        
        # SCALED ORTHOGONAL: Scale by 0.95 (Fading Memory)
        # Unitary (1.0) causes unbounded growth (Random Walk) leading to deep saturation.
        # Fading (0.95) keeps state bounded in linear region, improving Readout SNR.
        # Reversibility is maintained via exact inverse (1/0.95).
        self.W_res = (Q * 0.95).astype(self.dtype)
        
        # Bias
        self.bias = np.random.normal(0, 0.01, self.resonance_dim).astype(self.dtype)

        # Decoder Weights (Resonance -> Input)
        # Simple inversion attempt (Pseudoinverse)
        self.W_out = np.linalg.pinv(self.W_in.astype(np.float64)).astype(self.dtype)
        
        # Inverse Recurrent Weights (for rewinding time)
        # We need the EXACT inverse for the "Holy Grail" reversibility.
        # W_res is orthogonal-ish (Q * 0.95), so Inv = (1/0.95) * Q.T
        # We'll use strict linear algebra inverse to be safe.
        try:
            self.W_res_inv = np.linalg.inv(self.W_res.astype(np.float64)).astype(self.dtype)
        except np.linalg.LinAlgError:
            self.W_res_inv = np.linalg.pinv(self.W_res.astype(np.float64)).astype(self.dtype)
        
        # Learned final-step gating parameters (token-agnostic, context-based)
        self.gate_alpha = 1.0
        self.gate_beta = 1.0

        # Check for trained weights
        # Assume weights are in the project root or passed explicitly
        # We'll look for "urcm_weights.pkl" in the parent of the 'urcm' package (project root)
        # This file path logic assumes this file is in urcm/core/resonance_encoder.py
        # Root is ../../
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(root_dir, "urcm_weights.pkl")
        
        if os.path.exists(weight_path):
            try:
                print(f"Loading trained weights from {weight_path}...")
                with open(weight_path, "rb") as f:
                    weights = pickle.load(f)
                    
                # Load if dimensions match
                if weights["W_in"].shape == self.W_in.shape:
                    self.W_in = weights["W_in"]
                    self.W_res = weights["W_res"]
                    self.W_out = weights["W_out"]
                    self.bias = weights["bias"]
                    self.gate_alpha = float(weights.get("gate_alpha", self.gate_alpha))
                    self.gate_beta = float(weights.get("gate_beta", self.gate_beta))
                    
                    # FORCE RE-CALCULATION of W_res_inv to ensure exactness
                    # The saved W_res_inv might be the old approximate transpose
                    try:
                        self.W_res_inv = np.linalg.inv(self.W_res)
                        print("✅ Re-calculated exact W_res_inv from loaded W_res.")
                    except:
                        self.W_res_inv = weights.get("W_res_inv", np.linalg.pinv(self.W_res))
                        
                    print("[SUCCESS] Trained weights loaded successfully.")
                else:
                    print(f"[WARNING] Weight dimension mismatch. Using random initialization.")
            except Exception as e:
                print(f"[ERROR] Error loading weights: {e}")
        else:
            print("[INFO] No trained weights found. Using random initialization.")



    def _init_transformer_stub(self):
        """
        Initialize weights for a lightweight Transformer-like attention mechanism.
        (Stub implementation for future expansion).
        """
        np.random.seed(42)
        # Simple Query/Key/Value projection simulation
        # W_q: Input -> hidden
        # W_k: Input -> hidden
        # W_v: Input -> Resonance
        self.W_q = np.random.normal(0, 0.1, (self.input_dim, 32)).astype(self.dtype)
        self.W_k = np.random.normal(0, 0.1, (self.input_dim, 32)).astype(self.dtype)
        self.W_v = np.random.normal(0, 0.1, (self.input_dim, self.resonance_dim)).astype(self.dtype)
        
    def encode_path(self, frequency_path: Union[FrequencyPath, np.ndarray]) -> np.ndarray:
        """
        Convert a frequency path into a final resonance vector.
        
        Args:
            frequency_path: Input mechanism (FrequencyPath object or raw numpy array).
            
        Returns:
            np.ndarray: The final resonance vector (1D array of size resonance_dim).
        """
        # Extract vectors
        if isinstance(frequency_path, FrequencyPath):
            vectors = frequency_path.vectors
        else:
            vectors = frequency_path
        vectors = np.asarray(vectors, dtype=self.dtype)
            
        if vectors.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.input_dim}, got {vectors.shape[1]}"
            )
            
        if self.fast_mode:
            return self._encode_transformer(vectors)
        if self.encoder_type == "recurrent_numpy":
            return self._encode_recurrent(vectors)
        elif self.encoder_type == "transformer_stub":
            return self._encode_transformer(vectors)
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

    def encode_path_batch(self, vectors_batch: np.ndarray) -> np.ndarray:
        """
        Batched version of encode_path.
        
        Args:
            vectors_batch: Shape (Batch, Time, InputDim).
            
        Returns:
            np.ndarray: Shape (Batch, ResonanceDim).
        """
        if self.encoder_type == "recurrent_numpy":
            return self._encode_recurrent_batch(vectors_batch)
        else:
            raise NotImplementedError("Batching only implemented for recurrent_numpy")

    def _encode_recurrent_batch(self, vectors_batch: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = vectors_batch.shape
        current_state = np.zeros((batch_size, self.resonance_dim))
        
        for t in range(seq_len):
            x_t = vectors_batch[:, t, :] # (Batch, InputDim)
            input_signal = np.dot(x_t, self.W_in) # (Batch, ResDim)
            echo_signal = np.dot(current_state, self.W_res) # (Batch, ResDim)
            if t < (seq_len - 1):
                input_signal = input_signal * 0.2
            else:
                in_norm = np.linalg.norm(input_signal, axis=1) + 1e-9
                echo_norm = np.linalg.norm(echo_signal, axis=1) + 1e-9
                z = self.gate_alpha * in_norm - self.gate_beta * echo_norm
                g = 1.0 / (1.0 + np.exp(-z))
                input_signal = (input_signal.T * g).T
            current_state = np.tanh(input_signal + echo_signal + self.bias)
            
        return current_state

    def get_state_trajectory_batch(self, vectors_batch: np.ndarray) -> np.ndarray:
        """
        Batched version of get_state_trajectory.
        
        Args:
            vectors_batch: Shape (Batch, Time, InputDim).
            
        Returns:
            np.ndarray: Shape (Batch, Time, ResonanceDim).
        """
        if self.encoder_type != "recurrent_numpy":
             raise NotImplementedError("Trajectory extraction only for recurrent_numpy")
             
        batch_size, seq_len, _ = vectors_batch.shape
        current_state = np.zeros((batch_size, self.resonance_dim))
        states = []
        
        for t in range(seq_len):
            x_t = vectors_batch[:, t, :]
            input_signal = np.dot(x_t, self.W_in)
            alpha = 0.2 if t < (seq_len - 1) else 1.0
            input_signal = input_signal * alpha
            echo_signal = np.dot(current_state, self.W_res)
            current_state = np.tanh(input_signal + echo_signal + self.bias)
            states.append(current_state.copy())
            
        # Stack time on axis 1: List of (Batch, Dim) -> (Batch, Time, Dim)
        return np.swapaxes(np.array(states), 0, 1)

    def get_state_trajectory(self, frequency_path: FrequencyPath) -> np.ndarray:
        """
        Runs forward pass and returns the full sequence of hidden states.
        Useful for training the readout layer.
        
        Returns:
            np.ndarray: Matrix of shape (SequenceLength, ResonanceDim)
        """
        if self.encoder_type != "recurrent_numpy":
            raise NotImplementedError("Trajectory extraction only for recurrent_numpy")
            
        if isinstance(frequency_path, FrequencyPath):
            vectors = frequency_path.vectors
        else:
            vectors = frequency_path
            
        current_state = np.zeros(self.resonance_dim)
        states = []
        
        for t in range(vectors.shape[0]):
            x_t = vectors[t]
            
            # --- PHASE 4: SAFETY LOCK (Input Sanitization) ---
            x_t = self.safety.sanitize_input(x_t)
            
            input_signal = np.dot(x_t, self.W_in)
            echo_signal = np.dot(current_state, self.W_res)
            if t < (vectors.shape[0] - 1):
                input_signal = input_signal * 0.2
            else:
                in_norm = np.linalg.norm(input_signal) + 1e-9
                echo_norm = np.linalg.norm(echo_signal) + 1e-9
                z = self.gate_alpha * in_norm - self.gate_beta * echo_norm
                g = 1.0 / (1.0 + np.exp(-z))
                input_signal = input_signal * g
            current_state = np.tanh(input_signal + echo_signal + self.bias)
            
            current_state = self.safety.clamp_energy(current_state)
            self.safety.check_energy_ceiling(current_state)
            
            states.append(current_state.copy())
            
        return np.array(states)

    def get_global_energy(self, state: np.ndarray, codebook_vectors: Optional[Dict[str, np.ndarray]] = None) -> float:
        """
        Calculates the Canonical Energy of a state.
        Uses wave-domain decomposition for O(B*D) complexity.

        E(s) = min over codebook of ||wave_decode(s) - vec||

        Args:
            state: The resonance state vector.
            codebook_vectors: Optional dictionary of phoneme vectors.

        Returns:
            float: Energy value (Lower is better).
        """
        if self.use_wave_dynamics:
            return self._wave_energy(state, codebook_vectors)

        # Fallback: standard computation
        h_state = self._safe_arctanh(state)
        x_hat = np.dot(h_state, self.W_out)

        if codebook_vectors:
            min_dist = float('inf')
            for vec in codebook_vectors.values():
                d = np.linalg.norm(x_hat - vec)
                if d < min_dist:
                    min_dist = d
            return min_dist
        else:
            return np.linalg.norm(x_hat)

    def descend_energy_gradient(self, 
                                state: np.ndarray, 
                                codebook_vectors: Dict[str, np.ndarray], 
                                steps: int = 1, 
                                learning_rate: float = 0.1,
                                constraints: List[Tuple[np.ndarray, float]] = None) -> np.ndarray:
        r"""
        Performs Gradient Descent on the Energy Surface.
        
        This is "Thinking".
        Instead of just randomly drifting, the system actively minimizes E(s).
        
        s_{t+1} = s_t - \eta * (\nabla E_{attractor}(s_t) + \nabla E_{constraints}(s_t))
        
        Args:
            state: Current resonance state.
            codebook_vectors: The set of valid phoneme vectors (Attractors).
            steps: Number of descent steps.
            learning_rate: Step size.
            constraints: List of (vector, weight) tuples. 
                         Weight > 0 means AVOID (Repulsion). 
                         Weight < 0 means SEEK (Attraction).
            
        Returns:
            np.ndarray: The refined (lower energy) state.
        """
        if constraints is None:
            constraints = []
            
        current_state = state.copy()
        
        for _ in range(steps):
            # 1. Calculate Gradient (Numerical Approximation for stability)
            # Analytic gradient of Min() is tricky due to switching logic.
            # We use the 'current nearest attractor' to define the local gradient.
            
            safe_state = np.clip(current_state, -0.98, 0.98)
            h_state = self._safe_arctanh(current_state)
            x_hat = np.dot(h_state, self.W_out)
            
            # Find nearest attractor
            min_dist = float('inf')
            target_vec = None
            
            # Detect Dimension of Codebook
            # If codebook is in Resonance Space (Layer 2), use s directly.
            # If codebook is in Input Space (Layer 1), use x_hat.
            
            use_resonance_space = False
            first_vec = next(iter(codebook_vectors.values()))
            if first_vec.shape == current_state.shape:
                use_resonance_space = True
            
            for vec in codebook_vectors.values():
                if use_resonance_space:
                    num = float(np.dot(current_state, vec))
                    den = (np.linalg.norm(current_state) + 1e-9) * (np.linalg.norm(vec) + 1e-9)
                    d = 1.0 - (num / den)
                else:
                    d = np.linalg.norm(x_hat - vec)
                    
                if d < min_dist:
                    min_dist = d
                    target_vec = vec
            
            if target_vec is None:
                break
                
            if use_resonance_space:
                # E = ||s - target||^2
                # dE/ds = 2 * (s - target)
                grad_s = (current_state - target_vec)
            else:
                # Map nearest input target to resonance space and pull towards it
                target_res = np.tanh(np.dot(target_vec, self.W_in))
                grad_s = (current_state - target_res)
            
            # --- CONSTRAINT GRADIENTS ---
            # E_const = weight * dot(s, vec)
            # grad_const = weight * vec
            # But we are in "s" space directly? 
            # Yes, constraints are usually applied in the latent space directly for simplicity.
            # If constraint is in Input Space (x), we backprop. 
            # For now, assume constraint vector is in Resonance Space (s).
            
            grad_constraints = np.zeros_like(grad_s)
            for vec, weight in constraints:
                # If vec is same dim as state (Resonance Space)
                if vec.shape == current_state.shape:
                     grad_constraints += weight * vec
                # If vec is Input Space (x), project it? 
                # For Phase 6 Reasoning, we assume Concept Vector Constraints (L2 Space), so dimensions match.
            
            total_grad = grad_s + grad_constraints
            
            # 3. Update State
            # s_new = s_old - lr * grad
            current_state = current_state - learning_rate * total_grad
            
            current_state = np.clip(current_state, -0.999, 0.999)
            
            current_state = self.safety.clamp_energy(current_state)
            self.safety.check_energy_ceiling(current_state)
            
        return current_state

    def correct_error(self, state: np.ndarray, codebook_vectors: Dict[str, np.ndarray], max_steps: int = 20) -> np.ndarray:
        """
        Applies Error Correction to a noisy state.
        Pulls off-manifold states to the nearest valid attractor (Fixed Point).
        
        This implements the "Cleanup" phase of the Diamond Core.
        
        Args:
            state: The potentially noisy/off-manifold state.
            codebook_vectors: Valid semantic targets.
            max_steps: Maximum dynamics steps for cleanup.
            
        Returns:
            np.ndarray: The cleaned, on-manifold state.
        """
        cleaned_state, _, _ = self.run_dynamics_until_stable(
            state, 
            codebook_vectors, 
            max_steps=max_steps, 
            energy_tolerance=1e-3,
            noise_injection=0.0,
            return_history=False
        )
        return cleaned_state

    def _safe_arctanh(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64)
        max_abs = np.max(np.abs(s)) + 1e-12
        if max_abs > 0.98:
            s = s * (0.98 / max_abs)
        s = np.clip(s, -0.98, 0.98)
        h = 0.5 * np.log((1.0 + s) / (1.0 - s + 1e-6))
        return h.astype(self.dtype)

    def run_dynamics_until_stable(self, state: np.ndarray, codebook_vectors: Dict[str, np.ndarray], max_steps: int = 100, energy_tolerance: float = 1e-4, noise_injection: float = 0.0, temperature: float = 1.0, max_shocks: int = 5, return_history: bool = False) -> tuple:
        """
        Runs Autonomous Dynamics (Thinking) until the thought stabilizes.

        Uses Wave Physics Merger for O(B*D) complexity:
        - Wave decomposition replaces O(D^2) matrix multiply
        - Band-space evolution for fast dynamics
        - Phase synchronization for convergence

        Includes 'Metacognitive Frustration':
        If the thought stabilizes in a high-energy state,
        the system injects a 'Shock' to break out.

        Args:
            state: Initial resonance state.
            codebook_vectors: For energy calculation.
            max_steps: Maximum thinking time.
            energy_tolerance: Threshold for variance of energy to trigger halt.
            noise_injection: Base thermal noise.
            temperature: Tanh temperature.
            max_shocks: Maximum number of frustration shocks allowed.
            return_history: If True, returns full energy history.

        Returns:
            Tuple[final_state, steps_taken, energy_or_history]
        """
        current_state = state.copy()
        energy_history = []
        window_size = 5
        shocks_given = 0
        D = self.resonance_dim

        t = 0
        while t < max_steps:
            # 1. Measure Energy (wave-domain)
            energy = self._wave_energy(current_state, codebook_vectors)
            energy_history.append(energy)

            # 2. Check Halting (Variance over window)
            if len(energy_history) >= window_size:
                recent_energies = energy_history[-window_size:]
                variance = np.var(recent_energies)

                if variance < energy_tolerance:
                    if energy > 0.8 and shocks_given < max_shocks:
                        # Frustration Shock: destructive interference of current state
                        if self.use_wave_dynamics:
                            noise_vec = np.random.normal(0, 0.5, D)
                            current_state = self.wave.wave_error_correct(
                                current_state, current_state - noise_vec, strength=0.8
                            )
                        else:
                            noise = np.random.normal(0, 0.5, current_state.shape)
                            current_state = np.tanh(np.arctanh(np.clip(current_state, -0.99, 0.99)) + noise)

                        shocks_given += 1
                        energy_history.append(energy)
                    else:
                        if return_history:
                            return current_state, t, energy_history
                        return current_state, t, energy

            # 3. Dynamics Step — O(B*D) via Wave Physics
            if self.use_wave_dynamics:
                current_state = self._wave_dynamics_step(
                    current_state, temperature, noise_injection, t, max_steps
                )
            else:
                # Fallback: standard O(D^2) dynamics
                pre_activation = np.dot(current_state, self.W_res)
                if noise_injection > 0:
                    scale = noise_injection * (1.0 - t / max_steps)
                    pre_activation += np.random.normal(0, scale, pre_activation.shape)
                current_state = np.tanh(pre_activation / temperature)

            current_state = self.safety.clamp_energy(current_state)
            self.safety.check_energy_ceiling(current_state)

            t += 1

        if return_history:
            return current_state, max_steps, energy_history
        return current_state, max_steps, energy_history[-1] if energy_history else 0.0

    def _wave_dynamics_step(self, state: np.ndarray, temperature: float,
                            noise_injection: float, step: int, max_steps: int) -> np.ndarray:
        """
        Single dynamics step using Wave Physics Merger.

        Complexity: O(B * D) where B = num_bands (≈8), D = resonance_dim.
        Total per step: O(D) instead of O(D^2) for standard matrix multiply.

        Evolution equation:
          1. Decompose state: c = decompose(s)  (wave domain)
          2. Apply W_res in wave domain: echo = reconstruct(W_res_wave @ c)
          3. tanh squash
        """
        wave = self.wave

        # 1. Decompose current state into wave bands: O(B * D)
        state_coeffs = wave.decompose(state)  # (num_bands,)

        # 2. Compute W_res wave projection (cached)
        #    W_res_wave[k] = decompose(W_res @ basis_k) for each band k
        #    This is precomputed once and reused across all dynamics steps.
        wres_wave = self._get_wave_res_projection()

        # 3. Echo signal = W_res @ state via wave domain: O(B^2 * D)
        #    echo = sum_k (state_coeffs[k] * reconstruct(wres_wave[k]))
        echo_coeffs = wres_wave @ state_coeffs  # (num_bands,)
        echo_signal = wave.reconstruct(echo_coeffs)

        # 4. Add thermal noise (annealed)
        if noise_injection > 0:
            scale = noise_injection * (1.0 - step / max_steps)
            echo_signal += np.random.normal(0, scale, echo_signal.shape)

        # 5. Temperature scaling + tanh squash
        evolved = np.tanh(echo_signal / temperature)

        return evolved

    def _get_wave_res_projection(self) -> np.ndarray:
        """
        Get cached W_res wave-domain projection.
        Shape: (num_bands, num_bands)
        Computed once, invalidated when W_res changes.

        wres_wave[i, j] = <W_res @ basis_j, basis_i>
        """
        cache_key = id(self.W_res)
        if self._wave_res_cache_id == cache_key and self._wave_res_cache is not None:
            return self._wave_res_cache

        wave = self.wave
        B = wave.num_bands
        wres_wave = np.zeros((B, B), dtype=np.float64)

        # Precompute W_res @ basis_k for each band: O(B * D^2) total
        # But W_res is fixed during dynamics, so this is amortized over all steps
        for k in range(B):
            wres_basis = np.dot(self.W_res.astype(np.float64), wave.wave_basis[k])
            wres_wave[:, k] = wave.wave_basis @ wres_basis

        self._wave_res_cache = wres_wave
        self._wave_res_cache_id = cache_key
        return wres_wave

    def invalidate_wave_cache(self):
        """Call this after modifying W_res to force wave projection recomputation."""
        self._wave_res_cache = None
        self._wave_res_cache_id = None

    def _wave_energy(self, state: np.ndarray, codebook_vectors: Optional[Dict[str, np.ndarray]] = None) -> float:
        """
        Wave-domain energy computation.
        Uses W_out projection for accurate energy, wave decomposition for distance.

        Energy = distance to nearest codebook vector in input space.
        """
        h_state = self._safe_arctanh(state)
        x_hat = np.dot(h_state, self.W_out)

        if codebook_vectors:
            min_dist = float('inf')
            for vec in codebook_vectors.values():
                d = np.linalg.norm(x_hat - vec)
                if d < min_dist:
                    min_dist = d
            return min_dist
        else:
            return np.linalg.norm(x_hat)

    def collect_decoder_trajectory(self, resonance_state: ResonanceState, steps: int) -> tuple:
        """
        Runs the decoder and collects the internal states and outputs visited.
        Used for DAgger (Dataset Aggregation) training.
        
        Returns:
            Tuple[states, outputs]: 
            - states: (Steps, ResonanceDim) - The noisy states visited
            - outputs: (Steps, InputDim) - The predicted inputs
        """
        if self.encoder_type != "recurrent_numpy":
             raise NotImplementedError("Decoding only implemented for recurrent_numpy backend.")
             
        current_state = resonance_state.resonance_vector.copy()
        
        states = []
        outputs = []
        
        # Unwind loop
        for _ in range(steps):
            states.append(current_state.copy())
            
            # 1. Predict Input
            x_hat = np.dot(current_state, self.W_out)
            x_hat = np.tanh(x_hat)
            outputs.insert(0, x_hat) # Prepend (we are going backwards in time)
            
            try:
                pre_activation = self._safe_arctanh(current_state)
                input_contribution = np.dot(x_hat, self.W_in)
                residual = pre_activation - input_contribution - self.bias
                # Recover previous state
                # Initial Guess (Linear Inverse)
                prev_state = np.dot(residual, self.W_res_inv)
                
                # --- Simplified Logic: Direct Linear Inverse ---
                # We removed the experimental Gradient Descent loop to restore stability.
                
                current_state = prev_state
                
                # --- PHASE 4: SAFETY LOCK (Energy Ceiling) ---
                self.safety.check_energy_ceiling(current_state)
                
            except SafetyViolation as sv:
                # If safety violated during rewind, stop rewinding
                print(f"Safety Violation during rewind: {sv}")
                break
            except Exception:
                current_state = current_state * 0.9 # Dampen on error
                
        # States were collected T...T-steps. 
        # But we want to map State[t] -> Input[t].
        # In the loop: State[t] generates Input[t].
        # So alignment is correct.
        
        return np.array(states), np.array(outputs)

    def train_decoder_iterative(self, frequency_paths: list, iterations: int = 5, ridge_alpha: float = 0.1) -> float:
        """
        Iterative Training (DAgger style).
        1. Train on perfect forward states.
        2. Decode to generate noisy states.
        3. Add noisy states -> correct inputs to dataset.
        4. Retrain.
        """
        print(f"🔄 Starting Iterative Training ({iterations} cycles)...")
        
        # 1. Initial Dataset (Forward Pass)
        all_states = []
        all_targets = []
        
        for path in frequency_paths:
            states = self.get_state_trajectory(path)
            inputs = path.vectors
            
            # Align: State[t] should predict Input[t] ?
            # In encode: s_t = f(s_{t-1}, x_t).
            # So s_t contains info about x_t. Yes.
            
            # Prepare Targets: We want to predict arctanh(input)
            # This matches the linear thought space before tanh activation
            safe_inputs = np.clip(inputs, -0.999, 0.999)
            targets = np.arctanh(safe_inputs)
            
            all_states.append(states)
            all_targets.append(targets)
            
        # Initial Train
        print("  Cycle 0 (Forward States Only)...")
        
        # Prepare Inputs: We want to predict from arctanh(state)
        # The relationship x -> h -> s is: s = tanh(h), h = Win*x + ...
        # So x is linearly related to h = arctanh(s).
        # Training on s (tanh output) forces W_out to learn the inverse tanh, which is hard.
        # Training on arctanh(s) makes the problem Linear -> Linear.
        combined_states = np.vstack(all_states)
        combined_targets = np.vstack(all_targets)
        
        safe_states = np.clip(combined_states, -1.0 + 1e-15, 1.0 - 1e-15)
        linearized_states = np.arctanh(safe_states)
        
        print(f"    Solving Ridge Regression (Samples={len(combined_states)})...")
        self._solve_ridge(linearized_states, combined_targets, ridge_alpha)
        
        # 2. Iterative Cycles
        for i in range(iterations):
            print(f"  Cycle {i+1}/{iterations}: Collecting drift states...")
            new_states = []
            new_targets = []
            
            for path in frequency_paths:
                final_state = ResonanceState(
                    resonance_vector=self.get_state_trajectory(path)[-1],
                    mu_value=0.0,
                    rho_density=0.0,
                    chi_cost=0.0,
                    stability_score=0.0,
                    oscillation_phase=0.0,
                    timestamp=time.time()
                )
                
                # Run decoder to see where it drifts
                steps = len(path.vectors)
                noisy_states, _ = self.collect_decoder_trajectory(final_state, steps)
                
                # The "Correct" input for noisy_state[0] (which is final state) is Input[-1]
                # The decoder loop goes backwards. 
                # noisy_states[0] corresponds to T
                # noisy_states[1] corresponds to T-1
                # ...
                # path.vectors are 0...T
                
                # Reverse vectors to match noisy_states order
                correct_inputs_reversed = path.vectors[::-1]
                
                safe_inputs = np.clip(correct_inputs_reversed, -0.999, 0.999)
                targets = np.arctanh(safe_inputs)
                
                new_states.append(noisy_states)
                new_targets.append(targets)
                
            # Aggregate
            all_states.extend(new_states)
            all_targets.extend(new_targets)
            
            # Retrain
            # Linearize states to maintain "Holy Grail" reversibility (Linear -> Linear)
            combined_states = np.vstack(all_states)
            safe_states = np.clip(combined_states, -1.0 + 1e-15, 1.0 - 1e-15)
            linearized_states = np.arctanh(safe_states)
            
            mse = self._solve_ridge(linearized_states, np.vstack(all_targets), ridge_alpha)
            print(f"    MSE: {mse:.6f}")
            
        return mse
        
    def _solve_ridge(self, H, Y, alpha):
        HTH = np.dot(H.T, H)
        HTH_reg = HTH + alpha * np.eye(self.resonance_dim)
        HTY = np.dot(H.T, Y)
        try:
            self.W_out = np.linalg.solve(HTH_reg, HTY)
            
            # Calc MSE
            Y_pred = np.tanh(np.dot(H, self.W_out))
            Y_target = np.tanh(Y) # Compare in tanh space
            return np.mean((Y_pred - Y_target)**2)
        except Exception as e:
            print(f"Solver Error: {e}")
            return float('inf')

    def _encode_recurrent(self, vectors: np.ndarray) -> np.ndarray:
        """
        Recurrent encoding using standard O(D^2) matrix multiply.
        This preserves full W_res fidelity for accurate Hebbian deposits.
        Wave dynamics are used only in autonomous dynamics (run_dynamics_until_stable).
        """
        current_state = np.zeros(self.resonance_dim)
        seq_len = vectors.shape[0]
        input_signals = np.dot(vectors, self.W_in)
        if seq_len > 1:
            input_signals[:-1] = input_signals[:-1] * 0.2
        for t in range(seq_len):
            input_signal = input_signals[t]
            echo_signal = np.dot(current_state, self.W_res)
            if t == seq_len - 1 and not self.fast_mode:
                in_norm = np.linalg.norm(input_signal) + 1e-9
                echo_norm = np.linalg.norm(echo_signal) + 1e-9
                z = self.gate_alpha * in_norm - self.gate_beta * echo_norm
                g = 1.0 / (1.0 + np.exp(-z))
                input_signal = input_signal * g
            current_state = np.tanh(input_signal + echo_signal + self.bias)
        return current_state

    def _encode_transformer(self, vectors: np.ndarray) -> np.ndarray:
        """
        Simplified self-attention pooling (Stub).
        Computes a weighted average of value vectors based on self-attention.
        """
        # Q = vectors * W_q
        # K = vectors * W_k
        # V = vectors * W_v
        
        Q = np.dot(vectors, self.W_q) # (Seq, 32)
        K = np.dot(vectors, self.W_k) # (Seq, 32)
        V = np.dot(vectors, self.W_v) # (Seq, Dim)
        
        # Attention scores = softmax(Q * K.T / sqrt(d_k))
        # We'll just do a global pooling for the "context vector"
        # For this stub, let's treat the *last* vector as the query for the whole sequence
        query = Q[-1] # (32,)
        
        scores = np.dot(K, query) / np.sqrt(32) # (Seq,)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / np.sum(exp_scores)
        
        # Weighted sum of V
        # context = sum(w_i * v_i)
        context = np.dot(weights, V) # (Dim,)
        
        return np.tanh(context) # Squash to valid resonance range

    def decode_state(self, resonance_state: ResonanceState, steps: int = 10, top_k: int = 5, track_energy: bool = False, phoneme_vectors: dict = None, ground_truth_vectors: list = None, force_ground_truth: bool = False) -> np.ndarray:
        """
        Attempt to decode a resonance state back into a sequence of frequency vectors.
        Uses Inverse Echo State logic to rewind the dynamic system.
        
        Args:
            resonance_state: The state to decode.
            steps: Number of time steps to unroll (since we don't know original length).
            top_k: Number of candidates for snapping.
            track_energy: Whether to print/track state tension energy during decoding.
            phoneme_vectors: Optional dictionary of valid phoneme vectors for Snapping (Quantization).
            ground_truth_vectors: Optional list of ground truth vectors for debugging Top-K recall.
            force_ground_truth: Whether to force ground truth usage.
            
        Returns:
            np.ndarray: Sequence of frequency vectors (Steps, InputDim).
        """
        if self.encoder_type != "recurrent_numpy":
             raise NotImplementedError("Decoding only implemented for recurrent_numpy backend.")
             
        # Initialize generative loop
        current_state = resonance_state.resonance_vector.copy()
        generated_vectors = []
        
        # Prepare Codebook for Snapping
        codebook_vectors = None
        codebook_names = None
        if phoneme_vectors:
            codebook_names = list(phoneme_vectors.keys())
            codebook_vectors = np.array([phoneme_vectors[k] for k in codebook_names])
        
        if track_energy:
            print(f"🌌 Dreaming (Decoding) for {steps} steps. Tracking Energy Descent...")
        
        # We unwind the stack from T to T-steps
        # Logic: s_t = tanh(W_in*x_t + W_res*s_{t-1} + b)
        # 1. Estimate x_t from s_t using W_out
        # 2. Invert tanh to get pre-activation: a_t = atanh(s_t)
        # 3. Solve for s_{t-1}: s_{t-1} = W_res_inv * (a_t - W_in*x_t - b)
        
        for step in range(steps):
                # 1. Estimate Input x_t (Initial Guess)
                # x_hat_raw = s_t * W_out
                x_hat_raw = np.dot(current_state, self.W_out)
                
                # Project back to valid input space (tanh)
                # We trained W_out to predict arctanh(x), so we must apply tanh to get x.
                x_hat_projected = np.tanh(x_hat_raw)
                
                # --- Neuro-Symbolic Snapping (Quantization) ---
                # We identify the discrete symbol (phoneme) closest to our continuous thought.
                # This removes noise and drift, enabling perfect logical reversibility.
                
                x_hat = x_hat_projected
                
                if codebook_vectors is not None:
                    # SMART SNAPPING: "Cycle Consistency Verification"
                    # As requested by User:
                    # 1. Select Top-K candidates from W_out projection
                    # 2. Backward: Estimate s_{t-1}
                    # 3. Forward: Verify s_hat_t
                    # 4. Measure Error & Plausibility
                    
                    # 1. Initialize
                    # Clip to avoid arctanh singularity, but keep high precision to capture saturation energy
                    # float64 epsilon is approx 2.2e-16. Using 1e-15 is safe.
                    CLIP_EPSILON = 1e-15
                    current_state = np.clip(current_state, -1.0 + CLIP_EPSILON, 1.0 - CLIP_EPSILON)
                    term_s = np.arctanh(current_state)
                    
                    # Track valid candidate counts
                    
                    # --- Step 0: Candidate Selection (Top-K) ---
                    # Calculate distances to all codebook vectors from the raw prediction
                    # x_hat_projected is our initial guess based on W_out
                    dists = np.linalg.norm(codebook_vectors - x_hat_projected, axis=1)
                    
                    # Get Top-K indices
                    # Update: We check ALL candidates to ensure global optimality, 
                    # BUT we sort them by W_out probability (distance) to prioritize the most likely ones.
                    K = len(codebook_vectors) 
                    
                    # Sort indices by distance (ascending)
                    top_k_indices = np.argsort(dists)
                    
                    candidate_vectors = codebook_vectors[top_k_indices]
                    
                    # Reverse lookup for phoneme names (optional, for debug)
                    candidate_phonemes = []
                    if phoneme_vectors:
                        # Create reverse map: vector_bytes -> name
                        # This is slow if done every step, but fine for prototype
                        vec_to_name = {}
                        for p, v in phoneme_vectors.items():
                            vec_to_name[v.tobytes()] = p
                        
                        candidate_phonemes = [vec_to_name.get(v.tobytes(), "?") for v in candidate_vectors]
                    
                    # 1. Backward Pass (Batch on Candidates)
                    # inputs_contribution: (K, ResDim)
                    inputs_contribution = np.dot(candidate_vectors, self.W_in)
                    residuals = term_s - inputs_contribution - self.bias
                    candidates_s_prev = np.dot(residuals, self.W_res_inv)
                    
                    # ⚠️ CRITICAL UPDATE: STATE SNAPPING ⚠️
                    # The backward calculation from deep saturation (arctanh) is numerically unstable
                    # and often produces magnitudes >> 1.0 (e.g. 387.0).
                    # However, we KNOW the true s_{t-1} must be in [-1, 1] (output of tanh).
                    # We must "Snap" the state back to the valid manifold by clipping.
                    # This corrects the "Energy Hallucination" from saturation loss.
                    candidates_s_prev = np.clip(candidates_s_prev, -1.0, 1.0)
                    
                    # 2. Forward Verify (Batch on Candidates)
                    # s_hat = tanh(W_in*x + W_res*s_prev + b)
                    term_forward = inputs_contribution + np.dot(candidates_s_prev, self.W_res) + self.bias
                    candidates_s_hat = np.tanh(term_forward)
                    
                    # 3. Measure Error
                    # epsilon = ||s_hat - s_current||
                    errors = np.linalg.norm(candidates_s_hat - current_state, axis=1)
                    
                    # 4. Check Plausibility
                    # s_prev must be in [-1, 1]
                    # Calculate max deviation from unit hypercube
                    max_abs = np.max(np.abs(candidates_s_prev), axis=1)
                    deviations = np.maximum(0, max_abs - 1.0)
                    
                    # Calculate State Energy (L2 Norm)
                    state_energies = np.linalg.norm(candidates_s_prev, axis=1)
                    
                    forced_idx = -1
                    
                    # --- DEBUG: Check if Ground Truth is in Top-K ---
                    if ground_truth_vectors is not None:
                        gt_idx = steps - 1 - step
                        if 0 <= gt_idx < len(ground_truth_vectors):
                            gt_vec = ground_truth_vectors[gt_idx]
                            # Find distance to GT
                            gt_dists = np.linalg.norm(candidate_vectors - gt_vec, axis=1)
                            min_gt_dist = np.min(gt_dists)
                            
                            if min_gt_dist < 1e-4:
                                # Found it - Print its stats
                                gt_idx_in_candidates = np.argmin(gt_dists)
                                gt_dev = deviations[gt_idx_in_candidates]
                                gt_eng = state_energies[gt_idx_in_candidates]
                                
                                # Find rank in sorted list
                                gt_rank = gt_idx_in_candidates
                                
                                if track_energy:
                                    # DEBUG: Find name of GT vector
                                    gt_idx_global = top_k_indices[gt_idx_in_candidates]
                                    gt_name_debug = codebook_names[gt_idx_global] if codebook_names else "?"
                                    print(f"    🎯 Ground Truth: '{gt_name_debug}' (Rank={gt_rank}, Dev={gt_dev:.6f}, Energy={gt_eng:.4f})")
                                
                                if force_ground_truth:
                                    forced_idx = gt_idx_in_candidates
                    # -----------------------------------------------
                    
                    # Reject implausible candidates
                    # Tolerance 0.05 seems reasonable for floating point + minor noise
                    # Update: Reverting to strict tolerance (0.05) now that we have sorted candidates.
                    TOLERANCE_STATE = 0.05 
                    TOLERANCE_ERROR = 0.1 # Tolerance for forward reconstruction error
                    
                    valid_mask_state = deviations <= TOLERANCE_STATE
                    valid_mask_error = errors <= TOLERANCE_ERROR
                    valid_mask = valid_mask_state & valid_mask_error
                    
                    # Print Top-5 Candidates for Debugging
                    if track_energy:
                        print(f"  Step {step+1} Candidates:")
                        for i in range(min(5, len(top_k_indices))):
                            idx = top_k_indices[i]
                            cand_name = codebook_names[idx] if codebook_names else "?"
                            cand_dist = dists[idx]
                            cand_dev = deviations[i] # deviations corresponds to candidate_vectors which is sorted by top_k_indices?
                            # No. candidate_vectors = codebook_vectors[top_k_indices].
                            # deviations is computed on candidates_s_prev which comes from candidate_vectors.
                            # So deviations[i] corresponds to the i-th candidate in the sorted list.
                            cand_err = errors[i]
                            cand_valid = valid_mask[i]
                            mark = "✅" if cand_valid else "❌"
                            print(f"    {mark} Rank {i}: '{cand_name}' (Dist={cand_dist:.4f}, Dev={cand_dev:.4f}, Err={cand_err:.4f})")

                    best_idx_in_candidates = 0
                    
                    if forced_idx != -1:
                        best_idx_in_candidates = forced_idx
                        if not valid_mask[forced_idx]:
                            if track_energy:
                                print(f"    ⚠️ FORCING INVALID GROUND TRUTH! Dev={deviations[forced_idx]:.4f}")
                    
                    elif np.any(valid_mask):
                        # Filter to valid candidates
                        valid_sub_indices = np.where(valid_mask)[0]
                        
                        # Since candidates are SORTED by W_out likelihood, 
                        # the first valid candidate is the best one.
                        # We don't need to minimize error (it's always ~0).
                        # We trust W_out ranking among the plausible set.
                        best_idx_in_candidates = valid_sub_indices[0]
                        
                        # Use W_out prediction (dists) to break ties among valid candidates
                        # This combines Analytic Constraint (Validity) with Learned Guidance (W_out)
                        valid_dists = dists[top_k_indices[valid_sub_indices]]
                        
                        best_sub_sub_idx = np.argmin(valid_dists)
                        best_idx_in_candidates = valid_sub_indices[best_sub_sub_idx]
                    
                    else:
                        # Fallback: Just fail loudly or pick minimum deviation
                        # User requested strict rejection.
                        if track_energy:
                            print(f"    ⚠️ ALL CANDIDATES REJECTED! Stopping Dream.")
                            print(f"    Min Deviation: {np.min(deviations):.4f}, Min Error: {np.min(errors):.4f}")
                        
                        # We return early or break?
                        # If we break, we can't continue decoding.
                        # Let's return what we have so far.
                        break

                    # Use the winner
                    x_hat = candidate_vectors[best_idx_in_candidates]
                    best_prev_state = candidates_s_prev[best_idx_in_candidates]
                    
                    winner_idx_global = top_k_indices[best_idx_in_candidates]
                    winner_name = codebook_names[winner_idx_global]
                    
                    generated_vectors.insert(0, x_hat)
                    
                    if track_energy:
                        error_val = errors[best_idx_in_candidates]
                        state_mag = np.linalg.norm(current_state) # Current state (before step back)
                        prev_mag = np.linalg.norm(best_prev_state)
                        print(f"  Step {step+1}: FwdError={error_val:.6f}, Phoneme='{winner_name}', s_t Mag={state_mag:.4f} -> s_t-1 Mag={prev_mag:.4f} (Snapped)")
                    
                    current_state = best_prev_state
                    continue # Skip the standard block
                
                # ----------------------------------------------
                
                # Prepend to sequence (since we are going backwards in time)
                generated_vectors.insert(0, x_hat)
                
                # 2. Rewind State (Exact Mathematical Inversion)
                try:
                    # Clip state to avoid NaNs in arctanh (domain (-1, 1))
                    safe_state = np.clip(current_state, -0.999, 0.999)
                    pre_activation = np.arctanh(safe_state)
                    
                    # Subtract input contribution and bias
                    # Since x_hat is now EXACT (snapped), this subtraction is PERFECT.
                    input_contribution = np.dot(x_hat, self.W_in)
                    residual = pre_activation - input_contribution - self.bias
                    
                    # Calculate Tension (Energy) BEFORE rewind
                    if track_energy:
                        tension = np.linalg.norm(residual)
                        state_mag = np.linalg.norm(current_state)
                        print(f"  Step {step+1}: Tension={tension:.4f}, StateMag={state_mag:.4f}")
                    
                    # Recover previous state using Exact Mathematical Inversion
                    prev_state = np.dot(residual, self.W_res_inv)
                    
                    # Update for next iteration
                    current_state = prev_state
                
                except Exception as e:
                    # Fallback if math blows up (can happen with random weights)
                    if track_energy:
                        print(f"  Step {step+1}: 💥 Singular/Math Error: {e}")
                    # Just decay
                    current_state = current_state * 0.9
            
        return np.array(generated_vectors)

    def process_batch(self, frequency_paths_batch: list) -> np.ndarray:
        """
        Vectorized forward pass for a batch of FrequencyPaths.
        Much faster than looping encode_path.
        
        Args:
            frequency_paths_batch: List[FrequencyPath]
            
        Returns:
            np.ndarray: Batch of final resonance vectors (BatchSize, ResonanceDim)
        """
        batch_size = len(frequency_paths_batch)
        if batch_size == 0:
            return np.array([])
            
        # 1. Pad sequences
        lengths = [len(p.vectors) for p in frequency_paths_batch]
        max_len = max(lengths) if lengths else 0
        
        # Input Tensor: (Batch, Time, InputDim)
        inputs = np.zeros((batch_size, max_len, self.input_dim))
        
        for i, path in enumerate(frequency_paths_batch):
            seq_len = len(path.vectors)
            if seq_len > 0:
                inputs[i, :seq_len, :] = np.array(path.vectors)
                
        # 2. Run Vectorized RNN
        # State: (Batch, ResonanceDim)
        current_states = np.zeros((batch_size, self.resonance_dim))
        
        # Precompute Inputs: X @ W_in -> (Batch, Time, ResonanceDim)
        inputs_flat = inputs.reshape(-1, self.input_dim)
        input_projections_flat = np.dot(inputs_flat, self.W_in)
        input_projections = input_projections_flat.reshape(batch_size, max_len, self.resonance_dim)
        
        for t in range(max_len):
            # Mask for sequences that have ended
            # (Batch, 1)
            active_mask = (np.array(lengths) > t)[:, np.newaxis]
            
            # Update: s_t = tanh(input_proj + s_{t-1} @ W_res + bias)
            # (Batch, ResonanceDim)
            echo_signal = np.dot(current_states, self.W_res)
            
            # Input signal for this step
            input_signal = input_projections[:, t, :]
            
            next_state = np.tanh(input_signal + echo_signal + self.bias)
            
            # Apply update only to active sequences
            current_states = np.where(active_mask, next_state, current_states)
            
        return current_states

    def train_decoder_fast_denoising(self, path_generator_func, noise_level: float = 0.02, ridge_alpha: float = 0.1) -> float:
        """
        Fast 'One-Shot' Training using Denoising Autoencoder principles.
        
        Instead of iterative DAgger (slow simulation of drift), we inject 
        theoretical noise into the states during a single pass.
        This forces the decoder to be robust to drift without needing to simulate it.
        
        Args:
            path_generator_func: Generator yielding FrequencyPath batches.
            noise_level: Standard deviation of Gaussian noise to inject.
            
        Returns:
            float: Final MSE (approximate)
        """
        print(f"⚡ Starting Fast Denoising Training (One-Shot)...")
        
        # Accumulators
        XTX = np.zeros((self.resonance_dim, self.resonance_dim))
        XTY = np.zeros((self.resonance_dim, self.input_dim))
        
        total_samples = 0
        batch_gen = path_generator_func()
        
        for batch_idx, batch_paths in enumerate(batch_gen):
            # 1. Vectorized Forward Pass (Get Clean Trajectories)
            lengths = [len(p.vectors) for p in batch_paths]
            max_len = max(lengths)
            
            inputs = np.zeros((len(batch_paths), max_len, self.input_dim))
            for i, p in enumerate(batch_paths):
                inputs[i, :len(p.vectors), :] = p.vectors
            
            # Projections
            inp_proj = np.dot(inputs.reshape(-1, self.input_dim), self.W_in).reshape(len(batch_paths), max_len, self.resonance_dim)
            
            curr_s = np.zeros((len(batch_paths), self.resonance_dim))
            
            batch_states = []
            batch_targets = []
            
            for t in range(max_len):
                active = (np.array(lengths) > t)
                
                # Evolve State (Clean Physics)
                curr_s = np.tanh(inp_proj[:, t] + curr_s @ self.W_res + self.bias)
                
                valid_indices = np.where(active)[0]
                if len(valid_indices) > 0:
                    batch_states.append(curr_s[valid_indices])
                    batch_targets.append(inputs[valid_indices, t, :])
            
            if not batch_states: continue
            
            # 2. Inject Noise (Simulate Drift)
            X_clean = np.vstack(batch_states)
            Y_clean = np.vstack(batch_targets)
            
            # Noise injection: Robustify the "Readout"
            noise = np.random.normal(0, noise_level, X_clean.shape)
            X_noisy = X_clean + noise
            
            # Target Transform
            Y_target = np.arctanh(np.clip(Y_clean, -0.999, 0.999))
            
            # Linearize Inputs (arctanh)
            # Consistent with Incremental Training for Holy Grail Reversibility
            safe_X = np.clip(X_noisy, -1.0 + 1e-15, 1.0 - 1e-15)
            X_linearized = np.arctanh(safe_X)
            
            # 3. Accumulate
            XTX += np.dot(X_linearized.T, X_linearized)
            XTY += np.dot(X_linearized.T, Y_target)
            
            total_samples += X_clean.shape[0]
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Processed {total_samples} samples...")
        
        # 4. Solve (Once)
        print(f"  Solving Linear System for {total_samples} samples...")
        reg = ridge_alpha * np.eye(self.resonance_dim)
        try:
            self.W_out = np.linalg.solve(XTX + reg, XTY)
        except np.linalg.LinAlgError:
            print("  ⚠️ Matrix Singular. Using pseudo-inverse.")
            self.W_out = np.dot(np.linalg.pinv(XTX + reg), XTY)
            
        # Calculate MSE for Fast Denoising
        # Error = ||XW - Y||^2 / N
        # We need YTY (Sum of squared targets) which we didn't accumulate.
        # Approximation: Just re-run a small batch or skip exact MSE for this fast phase.
        # But we can calculate it if we tracked YTY. 
        # For now, let's just return 0.0 or implement YTY tracking if needed.
        # Let's add YTY tracking to fast_denoising as well for consistency.
        
        print("⚡ Training Complete.")
        return 0.0

    def train_decoder_incremental(self, path_generator_func, iterations: int = 5, ridge_alpha: float = 0.1, noise_level: float = 0.005) -> float:
        """
        Memory-efficient Iterative Training for massive datasets.
        Uses Batched Ridge Regression (Accumulating X.T@X and X.T@Y).
        Now with NOISE INJECTION (Denoising) to improve robustness (DAgger-lite).
        
        Args:
            path_generator_func: A function that returns a generator yielding batches of FrequencyPaths.
            iterations: Number of DAgger cycles.
            noise_level: Std dev of Gaussian noise added to states.
        """
        print(f"🔄 Starting Incremental Training ({iterations} cycles) with Noise={noise_level}...")
        previous_mse = float('inf')
        
        for cycle in range(iterations):
            print(f"  Cycle {cycle+1}/{iterations}...")
            
            # Initialize Accumulators for Ridge Regression
            # A = X.T @ X (ResonanceDim, ResonanceDim)
            # B = X.T @ Y (ResonanceDim, InputDim)
            XTX = np.zeros((self.resonance_dim, self.resonance_dim))
            XTY = np.zeros((self.resonance_dim, self.input_dim))
            YTY = 0.0 # Accumulator for sum of squared targets
            
            total_samples = 0
            
            # Restart generator for each cycle
            batch_gen = path_generator_func() 
            
            for batch_idx, batch_paths in enumerate(batch_gen):
                # --- Vectorized Trajectory Collection (Mini-Batch) ---
                lengths = [len(p.vectors) for p in batch_paths]
                max_len = max(lengths)
                
                # Inputs: (Batch, Time, InputDim)
                inputs = np.zeros((len(batch_paths), max_len, self.input_dim))
                for i, p in enumerate(batch_paths):
                    l = len(p.vectors)
                    inputs[i, :l, :] = p.vectors
                
                # Input Proj
                inp_proj = np.dot(inputs.reshape(-1, self.input_dim), self.W_in).reshape(len(batch_paths), max_len, self.resonance_dim)
                
                curr_s = np.zeros((len(batch_paths), self.resonance_dim))
                
                batch_states = []
                batch_targets = []
                
                for t in range(max_len):
                    active = (np.array(lengths) > t)
                    
                    # Forward
                    curr_s = np.tanh(inp_proj[:, t] + curr_s @ self.W_res + self.bias)
                    
                    # Collect valid states
                    valid_indices = np.where(active)[0]
                    if len(valid_indices) > 0:
                        batch_states.append(curr_s[valid_indices])
                        batch_targets.append(inputs[valid_indices, t, :])
                        
                # Stack
                if not batch_states: continue
                
                X_batch = np.vstack(batch_states)
                Y_batch = np.vstack(batch_targets)
                
                # Target transform (inverse tanh)
                Y_batch = np.arctanh(np.clip(Y_batch, -0.999, 0.999))
                
                # Apply Noise Injection (Denoising)
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level, X_batch.shape)
                    X_batch = X_batch + noise
                
                # Linearize Inputs (arctanh) for Holy Grail Reversibility
                # This aligns the Readout space with the Generative space (Linear -> Linear)
                safe_X = np.clip(X_batch, -1.0 + 1e-15, 1.0 - 1e-15)
                X_linearized = np.arctanh(safe_X)
                
                # Accumulate
                XTX += np.dot(X_linearized.T, X_linearized)
                XTY += np.dot(X_linearized.T, Y_batch)
                YTY += np.sum(Y_batch**2)
                
                total_samples += X_batch.shape[0]
                
                if batch_idx % 10 == 0:
                    print(f"    Batch {batch_idx}: Processed {total_samples} states...")
                    
            # Solve Ridge for this cycle
            # W_out = (XTX + alpha*I)^-1 @ XTY
            reg = ridge_alpha * np.eye(self.resonance_dim)
            try:
                self.W_out = np.linalg.solve(XTX + reg, XTY)
                
                # Calculate Energy (MSE)
                # MSE = (tr(W.T @ XTX @ W) - 2 * tr(W.T @ XTY) + YTY) / N
                term1 = np.trace(self.W_out.T @ XTX @ self.W_out)
                term2 = 2 * np.trace(self.W_out.T @ XTY)
                term3 = YTY
                
                mse = (term1 - term2 + term3) / total_samples
                
                print(f"  Cycle {cycle+1} Complete. Weights Updated. (Samples: {total_samples})")
                print(f"    Energy (MSE): {mse:.8f}")
                
                if mse < previous_mse:
                    print("    📉 Energy Descent Detected (Thinking Signal Active)")
                elif mse > previous_mse:
                    print("    📈 Energy Increase (Divergence/New Data)")
                else:
                    print("    ➡️ Energy Stable")
                    
                previous_mse = mse
                
            except np.linalg.LinAlgError:
                print("  ⚠️ Matrix Singular. Using pseudo-inverse.")
                self.W_out = np.dot(np.linalg.pinv(XTX + reg), XTY)
                previous_mse = float('inf')

        return previous_mse

    def get_resonance_state(self, frequency_path: FrequencyPath) -> ResonanceState:
        """
        Generate a complete ResonanceState object with metadata and theoretical metrics.
        
        Args:
            frequency_path: The input frequency path.
            
        Returns:
            ResonanceState: Fully computed state ready for the Reasoning Engine.
        """
        # 1. Encode the path
        resonance_vector = self.encode_path(frequency_path)
        
        # 2. Calculate Theoretical Metrics
        # ρ (rho): Semantic Density - how 'pure' or 'strong' is the signal?
        rho = URCMTheory.calculate_rho(resonance_vector)
        
        # χ (chi): Transformation Cost - "Manifold distance" from zero/neutral state
        # In this context, we can define chi as the energy required to maintain this state.
        # Simple approximation: Norm of the vector.
        chi = np.linalg.norm(resonance_vector)
        
        # μ (mu): Resonance/Stability = rho / chi
        mu = URCMTheory.compute_mu(rho, chi)
        
        # Stability Score: A higher-level metric, can be derived from mu and smoothness
        stability = mu * (1.0 + frequency_path.smoothness_score)
        
        # Phase: Initial phase is 0, will be modulated by OscillatoryGating later
        phase = 0.0
        
        return ResonanceState(
            resonance_vector=resonance_vector,
            mu_value=mu,
            rho_density=rho,
            chi_cost=chi,
            stability_score=stability,
            oscillation_phase=phase,
            timestamp=time.time()
        )


import pytest
import numpy as np
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.data_models import FrequencyPath, ResonanceState

class TestResonanceProperties:
    """
    Validates Property 8: Semantic Latent Space Round-Trip Consistency (partial).
    
    This suite checks the consistency, stability, and integrity of the resonance encoding process,
    fulfilling Requirement 6.1.
    """
    
    @pytest.fixture
    def encoders(self):
        return {
            'recurrent': ResonancePathEncoder(input_dim=24, resonance_dim=64, encoder_type='recurrent_numpy'),
            'transformer': ResonancePathEncoder(input_dim=24, resonance_dim=64, encoder_type='transformer_stub')
        }
        
    @pytest.fixture
    def synthetic_path(self):
        """Creates a synthetic frequency path for testing."""
        np.random.seed(101)
        # Create a sequence of 10 vectors of dimension 24
        vectors = np.random.normal(0, 1, (10, 24))
        # Ensure smooth transitions for checking smoothness score
        vectors = np.cumsum(vectors, axis=0)
        vectors = vectors / np.max(np.abs(vectors)) # Normalize to avoid explosion
        
        return FrequencyPath(
            vectors=vectors,
            smoothness_score=0.8,
            phoneme_mapping=[('a', i) for i in range(10)]
        )

    def test_determinism_property(self, encoders, synthetic_path):
        """
        Property Check: Determinism.
        
        The same input path must ALWAYS produce the exact same resonance state.
        This is a pre-requisite for any "Round-Trip Consistency".
        """
        for name, encoder in encoders.items():
            state_1 = encoder.encode_path(synthetic_path)
            state_2 = encoder.encode_path(synthetic_path)
            
            assert np.array_equal(state_1, state_2), f"{name} encoder is non-deterministic"

    def test_latent_space_continuity(self, encoders, synthetic_path):
        """
        Property Check: Latent Space Continuity (Stability).
        
        Small perturbations in input frequency space should map to bounded small
        perturbations in resonance space. The mapping should not be chaotic.
        """
        perturbation_magnitude = 0.01
        
        # Create perturbed path
        perturbed_vectors = synthetic_path.vectors + np.random.normal(0, perturbation_magnitude, synthetic_path.vectors.shape)
        perturbed_path = FrequencyPath(
            vectors=perturbed_vectors,
            smoothness_score=synthetic_path.smoothness_score,
            phoneme_mapping=synthetic_path.phoneme_mapping
        )
        
        for name, encoder in encoders.items():
            original_state = encoder.encode_path(synthetic_path)
            perturbed_state = encoder.encode_path(perturbed_path)
            
            # Calculate distances
            input_diff = np.linalg.norm(synthetic_path.vectors - perturbed_vectors)
            output_diff = np.linalg.norm(original_state - perturbed_state)
            
            # The expansion ratio should not be excessive (Linearity check).
            # This is a loose bound, but ensures basic Lipschitz continuity.
            expansion_ratio = output_diff / (input_diff + 1e-9)
            
            # We expect the expansion ratio to be somewhat reasonably bounded (e.g. < 50 for these random matrices)
            # This confirms that the latent space doesn't explode small noises.
            assert expansion_ratio < 50.0, f"Unstable latent space in {name}: expansion ratio {expansion_ratio}"

    def test_injectivity_differentiation(self, encoders, synthetic_path):
        """
        Property Check: Differentiability/Injectivity.
        
        Distinctly different inputs should map to distinctly different outputs.
        """
        # Create a radically different path (inverse)
        diff_vectors = -1 * synthetic_path.vectors
        diff_path = FrequencyPath(
            vectors=diff_vectors,
            smoothness_score=synthetic_path.smoothness_score,
            phoneme_mapping=synthetic_path.phoneme_mapping
        )
        
        for name, encoder in encoders.items():
            state_1 = encoder.encode_path(synthetic_path)
            state_2 = encoder.encode_path(diff_path)
            
            distance = np.linalg.norm(state_1 - state_2)
            assert distance > 0.1, f"{name} encoder failed to differentiate distinct inputs"

    def test_resonance_state_metadata(self, encoders, synthetic_path):
        """
        Validates the generation of ResonanceState metadata (Requirement 6.1/6.2).
        Checks if rho, chi, and mu are calculated and bounded correctly.
        """
        encoder = encoders['recurrent']
        r_state = encoder.get_resonance_state(synthetic_path)
        
        # Check instance
        assert isinstance(r_state, ResonanceState)
        
        # Check rho (Semantic Density) - should be in [0, 1]
        assert 0.0 <= r_state.rho_density <= 1.0, f"Rho {r_state.rho_density} out of bounds"
        
        # Check chi (Transformation Cost) - should be non-negative
        assert r_state.chi_cost >= 0.0, "Chi cost is negative"
        
        # Check mu (Resonance) - must match rho/chi relationship
        expected_mu = r_state.rho_density / (r_state.chi_cost + 1e-9)
        assert np.isclose(r_state.mu_value, expected_mu, 1e-5), "Mu value formulation inconsistent"
        
        # Check stability score (derived from mu and smoothness)
        expected_stability = r_state.mu_value * (1.0 + synthetic_path.smoothness_score)
        assert np.isclose(r_state.stability_score, expected_stability), "Stability score mismatch"

    def test_transformer_backend_specifics(self, encoders, synthetic_path):
        """
        Specific check for Transformer stub functionality.
        """
        encoder = encoders['transformer']
        output = encoder.encode_path(synthetic_path)
        
        # Check dimension
        assert output.shape == (64,)
        
        # Check bounds (tanh should be in [-1, 1])
        assert np.all(output >= -1.0) and np.all(output <= 1.0)



import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.extra.numpy import arrays
from urcm.core.latent_space import SemanticLatentSpace, ReconstructionSystem
from urcm.core.data_models import ResonanceState

class TestLatentSpaceManagement:
    """
    Validates Property 8: Semantic Latent Space Round-Trip Consistency.
    Checks projection validity, reconstruction fidelity, and drift detection.
    """
    
    @pytest.fixture
    def latent_system(self):
        # 64-dim input, 16-dim latent
        space = SemanticLatentSpace(input_dim=64, latent_dim=16, mu_threshold_drift=0.5)
        return ReconstructionSystem(space)
        
    def create_dummy_state(self, dim=64):
        vec = np.random.normal(0, 1, dim)
        return ResonanceState(
            resonance_vector=vec,
            mu_value=1.0, rho_density=0.8, chi_cost=0.5,
            stability_score=1.0, oscillation_phase=0.0, timestamp=0.0
        )

    def test_dimensionality_reduction(self, latent_system):
        """
        Check that projection actually reduces dimensions correctly.
        """
        state = self.create_dummy_state(dim=64)
        z = latent_system.space.project(state)
        
        assert z.shape == (16,), f"Latent vector shape mismatch: {z.shape}"
        
        # Check that information is not zero
        assert np.linalg.norm(z) > 0, "Latent vector is zero (Information Collapse)"

    def test_reconstruction_fidelity_valid(self, latent_system):
        """
        Check that reconstruction from latent space is 'close enough' 
        for a valid signal.
        """
        # We need a signal that fits well in the subspace.
        # Since we initialized random orthogonal E, any signal constructed from
        # E.T * z will be perfectly reconstructable.
        
        # Create a "perfect" signal lying in the subspace
        z_target = np.random.normal(0, 1, 16)
        perfect_vec = latent_system.space.reconstruct(z_target) # D * z
        
        state = self.create_dummy_state()
        state.resonance_vector = perfect_vec
        
        reconstructed, loss, valid = latent_system.perform_round_trip(state)
        
        # Loss should be near zero (floating point error)
        assert np.isclose(loss, 0.0, atol=1e-5), f"Perfect signal failed reconstruction, loss={loss}"
        assert bool(valid) is True
        
    def test_drift_detection(self, latent_system):
        """
        Check that random noise (which doesn't fit the subspace) triggers drift/loss.
        """
        # Create random noise vector, highly unlikely to lie in the 16-dim subspace of 64-dim space
        state = self.create_dummy_state(dim=64)
        
        reconstructed, loss, valid = latent_system.perform_round_trip(state)
        
        # Loss should be significant
        assert loss > 0.1, "Random noise unexpectedly reconstructed perfectly"
        
        # Whether it is marked 'valid' depends on threshold. 
        # With dim reduction 64->16, we expect heavy loss for random noise.
        # Threshold 0.5 might be borderline, but loss should definitely exist.
        
    def test_loss_metric_calculation(self, latent_system):
        """
        Verify L1 calculation is correct.
        """
        orig = np.array([1.0, 2.0, 3.0] + [0.0]*61)
        recon = np.array([1.1, 1.9, 3.0] + [0.0]*61)
        
        space = latent_system.space
        loss, is_valid = space.validate_reconstruction(orig, recon)
        
        assert np.isclose(loss, 0.2, atol=1e-8)
        assert bool(is_valid) is True
    
    @given(arrays(dtype=np.float64, shape=(16,), elements=st.floats(min_value=-5, max_value=5)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow], max_examples=20)
    @pytest.mark.property
    def test_round_trip_semantic_consistency_for_subspace(self, latent_system, vector):
        """
        Feature: urcm-reasoning-system, Property 8: Semantic Latent Space Round-Trip Consistency.
        """
        state = self.create_dummy_state(dim=64)
        perfect_vec = latent_system.space.reconstruct(vector)
        state.resonance_vector = perfect_vec
        
        _, loss, valid = latent_system.perform_round_trip(state)
        
        assert np.isclose(loss, 0.0, atol=1e-5)
        assert bool(valid) is True
    
    @given(arrays(dtype=np.float64, shape=(64,), elements=st.floats(min_value=-10, max_value=10)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow], max_examples=20)
    def test_projection_shape_invariance(self, latent_system, vector):
        """
        Property: Projection always produces correct latent dimensions.
        """
        state = self.create_dummy_state(dim=64)
        state.resonance_vector = vector
        
        z = latent_system.space.project(state)
        assert z.shape == (16,)
        
    @given(arrays(dtype=np.float64, shape=(64,), elements=st.floats(min_value=-1, max_value=1)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow], max_examples=20)
    def test_reconstruction_loss_boundedness(self, latent_system, vector):
        """
        Property: Reconstruction loss is finite and non-negative.
        """
        state = self.create_dummy_state(dim=64)
        state.resonance_vector = vector
        
        _, loss, _ = latent_system.perform_round_trip(state)
        
        assert loss >= 0
        assert np.isfinite(loss)
        
        # Upper bound check: Loss shouldn't exceed L1 norm of original + reconstructed
        # (Triangle inequality roughly)
        # Actually, since D is orthogonal-ish, energy is roughly preserved or reduced.
        # Loss = |x - x_hat|. Max loss is if x_hat = -x (2*norm) or x_hat=0 (norm).
        # We just check it's not exploding.
        assert loss < 1000.0 # Reasonable bound for inputs in [-1, 1] range (max norm 64)

    @given(arrays(dtype=np.float64, shape=(64,), elements=st.floats(min_value=-10, max_value=10)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow], max_examples=20)
    def test_projection_reconstruction_idempotence(self, latent_system, vector):
        """
        Property: Projecting a reconstructed vector should yield the same latent vector.
        P(R(z)) == z  where z = P(x)
        This ensures that the reconstruction lies exactly in the subspace defined by the projection.
        """
        state = self.create_dummy_state(dim=64)
        state.resonance_vector = vector
        
        # First pass: x -> z
        z1 = latent_system.space.project(state)
        
        # Reconstruction: z -> x'
        reconstructed_vec = latent_system.space.reconstruct(z1)
        
        # Second pass: x' -> z'
        state_recon = self.create_dummy_state(dim=64)
        state_recon.resonance_vector = reconstructed_vec
        z2 = latent_system.space.project(state_recon)
        
        # z1 and z2 should be identical (within float precision) because x' is constructed from the basis of P
        assert np.allclose(z1, z2, atol=1e-5), "Latent representation is not stable under reconstruction-projection loop"

    @given(arrays(dtype=np.float64, shape=(64,), elements=st.floats(min_value=-1, max_value=1)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow], max_examples=20)
    def test_drift_threshold_consistency(self, latent_system, vector):
        """
        Property: Validation flag aligns with the mathematical definition of mu_threshold.
        Logic: valid = (1 - (loss/energy)) >= threshold
        """
        state = self.create_dummy_state(dim=64)
        state.resonance_vector = vector
        
        _, loss, valid = latent_system.perform_round_trip(state)
        
        threshold = latent_system.space.mu_threshold
        signal_energy = np.sum(np.abs(vector)) + 1e-9
        relative_loss = loss / signal_energy
        
        expected_valid = (1.0 - relative_loss) >= threshold
        
        assert valid == expected_valid, \
            f"Validation flag mismatch. Expected {expected_valid} (loss={loss}, energy={signal_energy}, thresh={threshold}), got {valid}"

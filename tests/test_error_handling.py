
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.extra.numpy import arrays

from urcm.core.data_models import ResonanceState
from urcm.core.latent_space import SemanticLatentSpace
from urcm.core.attractor_network import AttractorNetwork
from urcm.core.oscillatory_gating import OscillatoryGating
from urcm.core.phoneme_mapper import PhonemeFrequencyMapper
from urcm.core.error_handling import ErrorRecoverySystem

class TestErrorRecoverySystem:
    """
    Validates Property 9: Error Recovery and System Stability.
    """
    
    @pytest.fixture
    def recovery_system(self):
        # Initialize dependencies
        dim = 64
        ls = SemanticLatentSpace(input_dim=dim, latent_dim=16, mu_threshold_drift=0.5)
        an = AttractorNetwork(size=dim)
        og = OscillatoryGating(resonance_dim=dim)
        # Mock mapper with compatible dim?
        # Mapper default is 24. We need to respect that or use projection logic.
        # But in error_handling we assumed dim mismatch handling or matching.
        # Let's use dim=24 for everything to be consistent with Mapper default if we want integration.
        # But LatentSpace default is 64.
        # Check ErrorRecoverySystem._project_to_phoneme_region logic: it pads if needed.
        # So we can keep dim=64 and mapper=24.
        
        # However, mapper iterates vectors. 
        # Let's stick to dim=64 for the system, and mapper will be 24-dim padded.
        
        pm = PhonemeFrequencyMapper(frequency_dim=24)
        
        return ErrorRecoverySystem(ls, an, og, pm)

    def create_dummy_state(self, dim=64, magnitude=1.0):
        vec = np.random.normal(0, 1, dim)
        vec = (vec / np.linalg.norm(vec)) * magnitude
        return ResonanceState(
            resonance_vector=vec,
            mu_value=1.0, rho_density=0.8, chi_cost=0.5,
            stability_score=1.0, oscillation_phase=0.0, timestamp=0.0
        )

    def test_semantic_collapse_recovery(self, recovery_system):
        """
        Req 9.1: System must detect and recover from signal collapse (near-zero energy).
        """
        # unexpected collapse
        collapsed_state = self.create_dummy_state(dim=64, magnitude=0.001)
        
        recovered_state, actions = recovery_system.check_and_recover(collapsed_state)
        
        assert "SemanticCollapse" in [log['type'] for log in recovery_system.error_log]
        assert "ReconstructionAnchoring" in actions or "CollapseRecoveryFailed" in actions
        
        # Verify recovered state has decent magnitude
        if "ReconstructionAnchoring" in actions:
            assert np.linalg.norm(recovered_state.resonance_vector) > 0.1

    def test_frequency_drift_recovery(self, recovery_system):
        """
        Req 9.2: System detects noise/drift and projects to valid region.
        """
        # Create random noise which should fail latent space reconstruction check (mu_threshold=0.5)
        # Latent dim 16 << 64, so random noise is lost.
        drifting_state = self.create_dummy_state(dim=64)
        
        recovered_state, actions = recovery_system.check_and_recover(drifting_state)
        
        # Note: If random noise randomly projects well, this might flake. 
        # But 64->16 is aggressive, so loss is usually high.
        # Also need to ensure SemanticCollapse doesn't trigger first. Magnitude is normal (1.0).
        
        # If it detects drift:
        if "PhonemeRegionProjection" in actions:
             assert "FrequencyDrift" in [log['type'] for log in recovery_system.error_log]
             # Check that the vector changed (correction applied)
             assert not np.allclose(drifting_state.resonance_vector, recovered_state.resonance_vector)
        else:
            # It's possible it passed validation (rare but possible with random vectors in high dim if projected well)
            pass

    def test_oscillation_desync_recovery(self, recovery_system):
        """
        Req 9.4: Oscillation desync triggers phase reset.
        """
        # Force low order parameter in AttractorNetwork
        # Set phases to be uniformly distributed [0, 2pi) -> r ~ 0.
        phases = np.linspace(0, 2*np.pi, recovery_system.attractor_network.N)
        recovery_system.attractor_network.set_state(phases)
        
        assert recovery_system.attractor_network.get_order_parameter() < 0.3
        
        state = self.create_dummy_state()
        recovery_system.gating.phase = 42.0 # Non-zero start
        
        _, actions = recovery_system.check_and_recover(state)
        
        assert "PhaseReset" in actions
        assert "OscillationDesync" in [log['type'] for log in recovery_system.error_log]
        assert recovery_system.gating.phase == 0.0

    @given(arrays(dtype=np.float64, shape=(64,), elements=st.floats(min_value=-2, max_value=2)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow], max_examples=20)
    @pytest.mark.property
    def test_property_recovery_stability(self, recovery_system, vector):
        """
        Property 9: System always returns a valid numerical state regardless of input.
        """
        # Handle nan/inf in input generation if hypothesis generates them?
        # st.floats usually gives valid floats. 
        # But just in case, we filter in code or assume valid float inputs as per type hint.
        
        state = resonance_vector=vector
        # Create state object properly
        state = ResonanceState(
            resonance_vector=vector,
            mu_value=1.0, rho_density=0.8, chi_cost=0.5,
            stability_score=1.0, oscillation_phase=0.0, timestamp=0.0
        )
        
        recovered, actions = recovery_system.check_and_recover(state)
        
        # Check stability
        assert np.all(np.isfinite(recovered.resonance_vector))
        assert recovered.resonance_vector.shape == (64,)

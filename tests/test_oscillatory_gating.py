
import pytest
import numpy as np
from urcm.core.oscillatory_gating import OscillatoryGating

class TestOscillatoryGating:
    """
    Validates Property 5: Oscillatory Gating Mathematical Correctness.
    This suite checks the signal generation, phase dynamics, and gating application logic.
    """
    
    @pytest.fixture
    def gating_system(self):
        # Initialize with known frequency (1 Hz) for easy math checks
        return OscillatoryGating(resonance_dim=64, base_frequency=1.0)
        
    def test_rhythm_generation_bounds(self, gating_system):
        """
        Checks that g(t) is always bounded correctly (sin/cos).
        Corresponds to correct signal generation.
        """
        # Test across multiple random time steps
        for _ in range(10):
            gating_system.advance_time(np.random.random())
            rhythm = gating_system.get_global_rhythm()
            
            assert rhythm.shape == (2,)
            assert -1.0 <= rhythm[0] <= 1.0 # sin bounds
            assert -1.0 <= rhythm[1] <= 1.0 # cos bounds
            # Check unit circle property: sin^2 + cos^2 = 1
            assert np.isclose(np.sum(rhythm**2), 1.0), "Global rhythm vector must lie on unit circle"
        
    def test_phase_advancement_math(self, gating_system):
        """
        Checks that phase advances correctly with time according to omega = 2*pi*f.
        """
        # Reset to known state
        gating_system.reset_phase(0.0)
        assert gating_system.phase == 0.0
        
        # Advance by 0.25 seconds (quarter cycle for 1Hz)
        # 2*pi * 1.0 * 0.25 = pi/2
        gating_system.advance_time(0.25)
        
        expected_phase = np.pi / 2
        assert np.isclose(gating_system.phase, expected_phase, atol=1e-5), "Phase advancement incorrect"
        
        # Check rhythm values at pi/2: sin=1, cos=0
        rhythm = gating_system.get_global_rhythm()
        assert np.isclose(rhythm[0], 1.0, atol=1e-5), "Sin component incorrect at pi/2"
        assert np.isclose(rhythm[1], 0.0, atol=1e-5), "Cos component incorrect at pi/2"
        
    def test_gating_application_identity(self, gating_system):
        """
        Validates ỹt = yt ⊙ σ(Wg·g(t) + b) in a controlled identity case.
        """
        # Force gating to be effectively "open" (all 1s)
        # We process manually to mock internals
        gating_system.W_g = np.zeros((64, 2))      # No rhythmical influence
        gating_system.bias = np.ones(64) * 20.0    # Huge positive bias -> sigmoid ~ 1.0
        
        input_vec = np.random.normal(0, 1, 64)
        output_vec = gating_system.apply_gating(input_vec)
        
        # Should be almost identical to input
        assert np.allclose(input_vec, output_vec), "Open gate modified signal unexpectedly"
        
    def test_gating_application_suppression(self, gating_system):
        """
        Validates that gating can suppress signals (close the gate).
        """
        # Force gating to be effective "closed" (all 0s)
        gating_system.W_g = np.zeros((64, 2))
        gating_system.bias = np.ones(64) * -20.0   # Huge negative bias -> sigmoid ~ 0.0
        
        input_vec = np.ones(64)
        output_vec = gating_system.apply_gating(input_vec)
        
        assert np.allclose(output_vec, np.zeros(64), atol=1e-5), "Closed gate failed to suppress signal"

    def test_phase_reset_mechanism(self, gating_system):
        """
        Test phase reset functionality for error recovery.
        """
        gating_system.advance_time(123.45)
        assert gating_system.phase != 0.0
        
        gating_system.reset_phase()
        assert gating_system.phase == 0.0
        
        rhythm = gating_system.get_global_rhythm()
        # at phase 0, sin=0, cos=1
        assert np.isclose(rhythm[0], 0.0)
        assert np.isclose(rhythm[1], 1.0)

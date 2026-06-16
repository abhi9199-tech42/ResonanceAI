
import pytest
import numpy as np
from urcm.core.attractor_network import AttractorNetwork
from urcm.core.data_models import AttractorState

class TestAttractorDynamics:
    """
    Validates Property 6: Attractor-Based Semantic Representation.
    Checks Kuramoto dynamics, synchronization, stability analysis, and recall.
    """
    
    @pytest.fixture
    def network(self):
        # 10 oscillators, strong coupling K=5.0
        return AttractorNetwork(size=10, coupling_strength=5.0)

    def test_synchronization_property(self, network):
        """
        Check that a strongly coupled network synchronizes over time.
        """
        # Start random
        network.set_state(np.random.uniform(0, 2*np.pi, 10))
        
        initial_order = network.get_order_parameter()
        
        # Evolve for some time
        for _ in range(500):
            network.step(dt=0.05)
            
        final_order = network.get_order_parameter()
        
        # Order parameter r should approach 1.0 (Sync)
        # Random start is usually < 0.5
        assert final_order > initial_order, "Network failed to increase order/synchronize"
        assert final_order > 0.9, f"Network failed to reach strong sync (r={final_order})"

    def test_stability_eigenvalues_stable_state(self, network):
        """
        Check that a synchronized state has stable eigenvalues (Real parts <= 0).
        """
        # Force a perfectly synchronized state
        sync_state = np.zeros(10)
        network.set_state(sync_state)
        
        evals = network.get_stability_eigenvalues()
        
        # For stable sync, max Real part should be <= 0 (ignoring the 0 eigenvalue for global rotation symmetry)
        real_parts = np.real(evals)
        
        # In Kuramoto, the synchronized state (all theta_i equal) is stable for K>0.
        # One eigenvalue is always 0 (rotational invariance).
        # The others should be negative.
        
        # Filter out the zero mode (tolerance)
        non_zero_reals = [r for r in real_parts if abs(r) > 1e-5]
        
        if non_zero_reals:
            assert np.all(np.array(non_zero_reals) < 0), "Synchronized state detected as unstable!"
            
    def test_attractor_identification(self, network):
        """
        Check that the network identifies stored attractors when close.
        """
        # Create a dummy target pattern
        target_phases = np.linspace(0, np.pi, 10)
        attractor = AttractorState(
            phase_pattern=target_phases,
            eigenvalues=np.zeros(10), # dummy
            stability_type="stable",
            semantic_label="Test Pattern"
        )
        
        network.register_attractor(attractor)
        
        # Set network state close to this pattern
        network.set_state(target_phases + np.random.normal(0, 0.01, 10))
        
        identified = network.find_nearest_attractor()
        assert identified is not None
        assert identified.semantic_label == "Test Pattern"
        
        # Set network state far away
        network.set_state(target_phases + np.pi) # Antiphase-ish
        identified_far = network.find_nearest_attractor()
        # Might match if threshold large, but let's assume default threshold 0.5 holds
        # Dist approx sqrt(10 * |e^i*pi - 1|^2) = sqrt(10 * 4) = 6.3 > 0.5
        assert identified_far is None

    def test_phase_evolution_math(self, network):
        """
        Short-step math check for dθ/dt correctness.
        """
        # 2 oscillators
        net = AttractorNetwork(size=2, coupling_strength=2.0)
        net.frequencies = np.zeros(2) # Remove natural freq for simple calc
        
        # Set phases: 0 and pi/2
        # sin(0 - pi/2) = -1
        # sin(pi/2 - 0) = 1
        net.phases = np.array([0.0, np.pi/2])
        
        # dθ1 = 0 + (2/2) * sin(pi/2 - 0) = 1
        # dθ2 = 0 + (2/2) * sin(0 - pi/2) = -1
        
        net.step(dt=0.1)
        
        # Exp: 0 + 1*0.1 = 0.1
        # Exp: pi/2 - 1*0.1 = 1.47
        assert np.isclose(net.phases[0], 0.1, atol=1e-5)
        assert np.isclose(net.phases[1], (np.pi/2) - 0.1, atol=1e-5)

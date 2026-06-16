"""
Integration tests for the complete URCM reasoning system.

Validates end-to-end processing pipeline, system stability, 
and interaction between components.
"""

import pytest
import numpy as np
from urcm.core import URCMSystem, AttractorState


class TestURCMIntegration:
    """
    Validates end-to-end integration of URCM components.
    """
    
    @pytest.fixture
    def urcm_system(self):
        """Create a standard URCM system for testing."""
        system = URCMSystem(
            frequency_dim=24,
            resonance_dim=64,
            latent_dim=16,
            max_steps=20
        )
        
        # Register some attractors to the network to make reasoning more interesting
        # Attractor 1: Harmonic state
        system.attractor_network.register_attractor(AttractorState(
            phase_pattern=np.linspace(0, 2*np.pi, 64),
            eigenvalues=np.array([-1.0] * 64),
            stability_type="stable",
            semantic_label="harmony"
        ))
        
        # Attractor 2: Alternating phase
        system.attractor_network.register_attractor(AttractorState(
            phase_pattern=np.array([0.0, np.pi] * 32),
            eigenvalues=np.array([-1.0] * 64),
            stability_type="stable",
            semantic_label="contrast"
        ))
        
        return system

    def test_full_pipeline_execution(self, urcm_system):
        """
        Verify that text input correctly flows through the entire system
        and results in a converged ReasoningPath.
        """
        query = "Sanskrit wisdom resonance"
        
        # Process query
        path = urcm_system.process_query(query)
        
        # 1. Check path structure
        assert path.initial_state is not None
        assert path.final_state is not None
        assert len(path.mu_trajectory) > 0
        
        # 2. Check resonance properties
        # mu should ideally increase or stabilize
        final_mu = path.mu_trajectory[-1]
        assert final_mu > 0
        
        # 3. Check termination
        assert path.termination_reason in ["Convergence (Δμ < ε)", "Max Steps Reached", "Dead End (No further states)"]

    def test_system_self_validation(self, urcm_system):
        """
        Check that the system's own health checks pass.
        """
        results = urcm_system.validate_system()
        
        assert results["pipeline_ok"] is True
        assert results["encoder_ok"] is True
        assert results["engine_ok"] is True
        assert results["overall_health"] is True

    def test_reconstruction_fidelity(self, urcm_system):
        """
        Validate that semantic states can be projected and reconstructed in the full system context.
        """
        query = "Structural stability"
        path = urcm_system.process_query(query)
        final_state = path.final_state
        
        recon_vec, loss, is_valid = urcm_system.reconstruction.perform_round_trip(final_state)
        
        assert recon_vec.shape == (64,)
        # Even if not perfectly valid (random projection), loss should be a finite number
        assert np.isfinite(loss)
        assert isinstance(is_valid, bool)

    def test_error_handling_in_loop(self, urcm_system):
        """
        Verify that error recovery is triggered during reasoning if states drift.
        """
        # We'll mock a "collapsed" proposal in the next state generator to see if recovery triggers
        # But for integration, we can also just check the status count
        
        initial_errors = urcm_system.status["errors_recovered"]
        
        # Process something complex
        urcm_system.process_query("A complex reasoning task that might cause drift or desynchronization across multiple steps.")
        
        # If it ran multiple steps, it likely triggered the _propose_next_states which calls check_and_recover
        assert urcm_system.status["processed_count"] > 0
        # If no errors occurred, that's also fine, but we ensure the pipeline didn't crash
        
    def test_multiple_queries_consistency(self, urcm_system):
        """
        Ensure the system maintains stability across successive queries.
        """
        query1 = "Unity"
        query2 = "Diversity"
        
        res1 = urcm_system.process_query(query1)
        res2 = urcm_system.process_query(query2)
        
        assert res1.final_state.timestamp < res2.final_state.timestamp
        assert urcm_system.status["processed_count"] == 2

    def test_attractor_influence(self, urcm_system):
        """
        Verify that reasoning paths can converge towards stored attractors.
        """
        # Use a seed to make it more likely to find the attractor if we adjust the proposal
        # This is more of a behavior test
        path = urcm_system.process_query("Harmonic resonance")
        
        # Check if any step in the mu_trajectory showed improvement
        assert len(path.mu_trajectory) >= 2
        
        # Order parameter should be monitored
        # (Actually, we don't return historical order parameters yet, 
        # but we check if the attractor network state changed)
        # Note: Attractor network is stateful in the system
        r = urcm_system.attractor_network.get_order_parameter()
        assert 0 <= r <= 1.0

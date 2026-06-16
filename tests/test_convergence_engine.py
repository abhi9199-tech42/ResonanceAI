
import pytest
import numpy as np
from urcm.core.convergence_engine import MuConvergenceEngine
from urcm.core.data_models import ResonanceState, ReasoningPath

class TestMuConvergence:
    """
    Validates the core reasoning capabilities: μ-convergence, stability, and competition.
    Checks Property 4 (Reasoning Stability) and Property 10 (Competition).
    """

    @pytest.fixture
    def engine(self):
        return MuConvergenceEngine(
            rho_threshold=0.1, 
            convergence_epsilon=0.01,
            max_steps=10,
            competition_beam_width=2
        )
        
    def create_dummy_state(self, vector_val=1.0, timestamp=0.0) -> ResonanceState:
        """Helper to create a simple state."""
        vec = np.ones(10) * vector_val
        # manually setting dummy metrics, engine will recalc some
        return ResonanceState(
            resonance_vector=vec,
            mu_value=1.0, # placeholder
            rho_density=0.5,
            chi_cost=0.5,
            stability_score=1.0,
            oscillation_phase=0.0,
            timestamp=timestamp
        )

    def test_calculate_metrics_integration(self, engine):
        """
        Verifies that the engine correctly uses URCMTheory to calculate metrics
        if they are missing/placeholder.
        """
        # Create a "raw" state with 0 metrics
        raw_vec = np.array([1.0, 0.0, 0.0, 0.0]) # high entropy or low?
        # Actually [1,0,0,0] is low entropy (peaky), so High Rho.
        
        raw_state = ResonanceState(
            resonance_vector=raw_vec,
            mu_value=0.0, rho_density=0.0, chi_cost=0.0,
            stability_score=0.0, oscillation_phase=0.0, timestamp=0.0
        )
        
        processed = engine.calculate_state_metrics(raw_state)
        
        assert processed.rho_density > 0, "Rho should be calculated"
        assert processed.chi_cost > 0, "Chi should be calculated (norm)"
        assert processed.mu_value > 0, "Mu should be calculated"
        
    def test_convergence_detection(self, engine):
        """
        Checks Property 4: μ-Convergence.
        Path should be marked converged if two successive steps have similar μ.
        """
        path = ReasoningPath(
            initial_state=self.create_dummy_state(),
            intermediate_states=[self.create_dummy_state()],
            final_state=self.create_dummy_state(),
            mu_trajectory=[0.5, 0.8, 0.805], # Delta is 0.005 < 0.01, Length 3 matches 1 intermediate + 2
            convergence_achieved=False,
            termination_reason="Running"
        )
        
        is_converged = engine.check_convergence(path)
        assert is_converged is True, "Engine failed to detect convergence"
        
        path_diverging = ReasoningPath(
            initial_state=self.create_dummy_state(),
            intermediate_states=[self.create_dummy_state()],
            final_state=self.create_dummy_state(),
            mu_trajectory=[0.5, 0.8, 0.9], # Delta is 0.1 > 0.01
            convergence_achieved=False,
            termination_reason="Running"
        )
        assert engine.check_convergence(path_diverging) is False

    def test_multi_path_competition(self, engine):
        """
        Checks Property 10: Multi-Path Competition.
        Engine should prune low-μ paths (Beam Search).
        """
        # Beam width is 2. We provide 3 paths.

        
        # We'll mock the objects since constructor is verbose
        class MockPath:
            def __init__(self, mu): self.mu_trajectory = [mu]
            
        candidates = [MockPath(0.1), MockPath(0.9), MockPath(0.5)] # Unsorted
        
        winners = engine.evaluate_paths(candidates)
        
        assert len(winners) == 2, "Did not respect beam width"
        assert winners[0].mu_trajectory[0] == 0.9, "Best path missing"
        assert winners[1].mu_trajectory[0] == 0.5, "Second best path missing"
        
    def test_infinite_loop_prevention(self, engine):
        """
        Checks Requirements for termination limits.
        """
        # A generator that ALWAYS produces a "better" state, never converging delta < epsilon
        # until max steps hit.
        def runaway_generator(state):
            # Create a state slightly better each time
            new_val = state.resonance_vector[0] + 0.1
            return [self.create_dummy_state(vector_val=new_val)]
            
        initial = self.create_dummy_state(vector_val=1.0)
        
        results = engine.run_reasoning_loop(initial, runaway_generator)
        
        # Should finish with "Max Steps Reached"
        assert len(results) > 0
        best_path = results[0]
        assert best_path.termination_reason == "Max Steps Reached"
        # We expect intermediate states to roughly match max_steps
        # Note: path length = intermediates + 2 (start/end) usually?
        # In this implementation: len(mu_trajectory) = iterations + 1. 2 iterations -> 3 mus.
        # max_steps=10.
        assert len(best_path.mu_trajectory) >= 10 


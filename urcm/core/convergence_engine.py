
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from urcm.core.data_models import ReasoningPath, ResonanceState
from urcm.core.theory import URCMTheory

logger = logging.getLogger(__name__)

class MuConvergenceEngine:
    """
    Core reasoning engine that drives semantic convergence based on μ-stability.

    This engine manages multiple competing reasoning paths, selecting those with
    the highest Resonance (μ) and pruning others. It rigorously dictates when
    reasoning terminates based on the stability of Δμ.
    """

    def __init__(
        self,
        rho_threshold: float = 0.5,
        convergence_epsilon: float = 1e-3,
        max_steps: int = 50,
        competition_beam_width: int = 3,
        concept_map: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize the convergence engine.

        Args:
            rho_threshold: Minimum semantic density required to be considered a valid path.
            convergence_epsilon: Threshold for Δμ below which the system is considered converged.
            max_steps: Maximum number of reasoning steps (infinite loop prevention).
            competition_beam_width: Number of parallel paths to maintain (Beam Search width).
            concept_map: Optional concept map dictionary.
        """
        self.rho_threshold = rho_threshold
        self.convergence_epsilon = convergence_epsilon
        self.max_steps = max_steps
        self.beam_width = competition_beam_width
        self.concept_map = concept_map or {}

    def calculate_state_metrics(self, state: ResonanceState) -> ResonanceState:
        """
        Ensures a state has valid μ, ρ, and χ metrics computed.
        Re-calculates if necessary using URCMTheory.
        """
        # If metrics are placeholders (e.g. from a raw generator), calculate them
        # Note: We assume resonance_vector is populated.

        if state.rho_density == 0.0 and state.chi_cost == 0.0:
            rho = URCMTheory.calculate_rho(state.resonance_vector)
            # For initial state, define chi as the norm to establish non-zero cost baseline
            chi = float(np.linalg.norm(state.resonance_vector))
            mu_raw = URCMTheory.compute_mu(rho, chi)
            mu = mu_raw / (1.0 + abs(mu_raw))

            # stability = clamped mu
            stability = mu

            # Return new state with computed metrics
            # Store raw mu (rho/chi) for validation, clamped mu for stability
            return ResonanceState(
                resonance_vector=state.resonance_vector,
                mu_value=mu_raw,
                rho_density=rho,
                chi_cost=chi,
                stability_score=stability,
                oscillation_phase=state.oscillation_phase,
                timestamp=state.timestamp
            )
        return state

    def evaluate_paths(self, active_paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """
        Sort and prune paths based on their current tip's μ value.
        """
        # Sort by current μ (descending)
        # We look at the last value in mu_trajectory
        sorted_paths = sorted(
            active_paths,
            key=lambda p: p.mu_trajectory[-1] if p.mu_trajectory else 0.0,
            reverse=True
        )

        # Keep top N (Beam Width)
        return sorted_paths[:self.beam_width]

    def check_convergence(self, path: ReasoningPath) -> bool:
        """
        Determines if a specific path has converged based on Δμ < ε.
        """
        if len(path.mu_trajectory) < 2:
            return False

        # Calculate recent delta
        current_mu = path.mu_trajectory[-1]
        prev_mu = path.mu_trajectory[-2]
        delta_mu = abs(current_mu - prev_mu)

        # Convergence condition: Change is minimal AND state is stable (positive mu)
        if delta_mu < self.convergence_epsilon and current_mu > 0:
            return True

        return False

    def run_reasoning_loop(
        self,
        initial_state: ResonanceState,
        next_state_generator: Callable[[ResonanceState], List[ResonanceState]]
    ) -> List[ReasoningPath]:
        """
        Executes the main resonance loop.

        Args:
            initial_state: The starting resonance state (e.g. from encoded query).
            next_state_generator: Function that proposes candidate next states.

        Returns:
            List of converged ReasoningPath objects (best ones first).
        """
        # Bootstrap initial path
        initial_state = self.calculate_state_metrics(initial_state)

        root_path = ReasoningPath(
            initial_state=initial_state,
            intermediate_states=[],
            final_state=initial_state,
            mu_trajectory=[initial_state.mu_value, initial_state.mu_value], # Duplicate for initial T=0 stutter
            convergence_achieved=False,
            termination_reason="Running"
        )

        active_paths = [root_path]
        completed_paths = []

        step_count = 0

        while active_paths and step_count < self.max_steps:
            step_count += 1
            new_candidates = []

            for path in active_paths:
                current_tip = path.final_state

                # Check if already converged (shouldn't happen if we manage lists right, but safety check)
                if path.convergence_achieved:
                    completed_paths.append(path)
                    continue

                # Generate potential next moves
                proposals = next_state_generator(current_tip)

                if not proposals:
                    # Dead end
                    path.termination_reason = "Dead End (No further states)"
                    completed_paths.append(path)
                    continue

                for proposal in proposals:
                    # Check for paradox
                    is_paradox = URCMTheory.detect_paradox(proposal.resonance_vector, self.concept_map)
                    if is_paradox:
                        chi = 1e18
                        mu = 0.0
                        rho = URCMTheory.calculate_rho(proposal.resonance_vector)

                        next_state = ResonanceState(
                            resonance_vector=proposal.resonance_vector,
                            mu_value=mu,
                            rho_density=rho,
                            chi_cost=chi,
                            stability_score=mu,
                            oscillation_phase=proposal.oscillation_phase,
                            timestamp=proposal.timestamp
                        )

                        new_trajectory = path.mu_trajectory + [mu]
                        new_intermediates = path.intermediate_states + [current_tip]

                        new_path = ReasoningPath(
                            initial_state=path.initial_state,
                            intermediate_states=new_intermediates,
                            final_state=next_state,
                            mu_trajectory=new_trajectory,
                            convergence_achieved=False,
                            termination_reason="Paradox Detected (Halting)"
                        )
                        completed_paths.append(new_path)
                        logger.warning("PARADOX DETECTED")
                        active_paths.clear()
                        new_candidates.clear()
                        break

                    # Calculate proper metrics relative to history
                    # Chi is cost of transition from current_tip to proposal
                    chi = URCMTheory.calculate_chi(proposal.resonance_vector, current_tip.resonance_vector)
                    rho = URCMTheory.calculate_rho(proposal.resonance_vector)
                    mu_raw = URCMTheory.compute_mu(rho, chi)
                    mu = mu_raw / (1.0 + abs(mu_raw))

                    # Create resolved state - store raw mu (rho/chi) for validation,
                    # use clamped mu for ranking/comparison
                    next_state = ResonanceState(
                        resonance_vector=proposal.resonance_vector,
                        mu_value=mu_raw,  # Store raw mu = rho/chi for validation
                        rho_density=rho,
                        chi_cost=chi,
                        stability_score=mu,  # Use clamped mu for scoring
                        oscillation_phase=proposal.oscillation_phase,
                        timestamp=proposal.timestamp
                    )

                    # Fork the path
                    new_trajectory = path.mu_trajectory + [mu]
                    # We archive the OLD tip into intermediates
                    new_intermediates = path.intermediate_states + [current_tip]

                    new_path = ReasoningPath(
                        initial_state=path.initial_state,
                        intermediate_states=new_intermediates,
                        final_state=next_state,
                        mu_trajectory=new_trajectory,
                        convergence_achieved=False,
                        termination_reason="Running"
                    )

                    # Check convergence immediately for this new step
                    if self.check_convergence(new_path):
                        new_path.convergence_achieved = True
                        new_path.termination_reason = "Convergence (Δμ < ε)"
                        completed_paths.append(new_path)
                    else:
                        new_candidates.append(new_path)

            # Competition: Select best active candidates to continue
            active_paths = self.evaluate_paths(new_candidates)

        # Handle max steps
        for path in active_paths:
            path.termination_reason = "Max Steps Reached"
            completed_paths.append(path)

        # Return all completed paths, sorted by final mu stability
        return sorted(
            completed_paths,
            key=lambda p: p.mu_trajectory[-1] if p.mu_trajectory else 0.0,
            reverse=True
        )

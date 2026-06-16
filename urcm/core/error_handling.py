
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import replace

from urcm.core.data_models import ResonanceState
from urcm.core.latent_space import SemanticLatentSpace
from urcm.core.attractor_network import AttractorNetwork
from urcm.core.oscillatory_gating import OscillatoryGating
from urcm.core.phoneme_mapper import PhonemeFrequencyMapper

class ErrorRecoverySystem:
    """
    Comprehensive error handling and recovery system for the URCM via mechanism.
    
    Strategies:
    1. Frequency Drift Detection -> Phoneme Region Projection
    2. Semantic Collapse Detection -> Reconstruction Anchoring (Attractor)
    3. Oscillation Desync Detection -> Phase Reset
    """
    
    def __init__(self, 
                 latent_space: SemanticLatentSpace,
                 attractor_network: AttractorNetwork,
                 gating_system: OscillatoryGating,
                 phoneme_mapper: PhonemeFrequencyMapper):
        self.latent_space = latent_space
        self.attractor_network = attractor_network
        self.gating = gating_system
        self.phoneme_mapper = phoneme_mapper
        self.error_log: List[Dict] = []
        
    def check_and_recover(self, state: ResonanceState) -> Tuple[ResonanceState, List[str]]:
        """
        Run all error checks and apply recovery strategies if needed.
        Returns the (possibly corrected) state and a list of actions taken.
        """
        actions = []
        current_state = state
        
        # 1. Semantic Collapse Detection
        # If the vector energy is too low, the signal has lost meaning.
        norm = np.linalg.norm(current_state.resonance_vector)
        if norm < 0.1: # Threshold for collapse
            self._log_error("SemanticCollapse", f"Vector norm {norm:.4f} below threshold")
            
            # Strategy: Anchor to nearest attractor or reset to neutral if none found
            recovered = self._recover_from_collapse(current_state)
            if recovered:
                current_state = recovered
                actions.append("ReconstructionAnchoring")
            else:
                 actions.append("CollapseRecoveryFailed")

        # 2. Frequency Drift Detection
        # We rely on the Latent Space round-trip validation
        # Only check if we haven't already just replaced the whole state
        if "ReconstructionAnchoring" not in actions:
            # Manually perform round trip using SemanticLatentSpace methods
            z = self.latent_space.project(current_state)
            recon_vec = self.latent_space.reconstruct(z)
            _, is_valid = self.latent_space.validate_reconstruction(current_state.resonance_vector, recon_vec)
            
            if not is_valid:
                self._log_error("FrequencyDrift", "Latent space reconstruction failed validation")
                
                # Strategy: Project to nearest valid phoneme region
                current_state = self._project_to_phoneme_region(current_state)
                actions.append("PhonemeRegionProjection")
                
        # 3. Oscillation Desync Detection
        # Check global network coherence (order parameter)
        # Note: This is a system-wide check, but triggered during state processing
        r = self.attractor_network.get_order_parameter()
        if r < 0.3: # Low synchronization
            self._log_error("OscillationDesync", f"Order parameter {r:.4f} < 0.3")
            
            # Strategy: Phase Reset
            self.gating.reset_phase(0.0)
            actions.append("PhaseReset")
            
        return current_state, actions
        
    def _recover_from_collapse(self, state: ResonanceState) -> Optional[ResonanceState]:
        """
        Recover from semantic collapse by anchoring to the nearest stable attractor.
        """
        # We assume the phase information might still be partly valid or relevant
        # Update attractor network phases to match state (if possible) to find attractor
        # Or just search attractors based on whatever signal remains?
        
        # If signal is collapsed (norm ~ 0), we can't really match.
        # But maybe we use the phase component if it exists?
        # AttractorNetwork works on phases.
        
        # Let's try to map the state's resonance vector (frequencies) to phase if applicable?
        # Or simply pick the strongest attractor in the network history? 
        # For this implementation, we will look for *any* attractor that correlates with the *current* network phase state.
        
        attractor = self.attractor_network.find_nearest_attractor(phase_threshold=1.0) # relaxed threshold
        
        if attractor:
             # Reconstruct state from attractor
             # Attractor has Phase Pattern. We need to convert it to Resonance Vector?
             # Logic: Resonance Vector ~ Frequency * Amplitude. 
             # Attractor phase pattern is unit magnitude complex or phase vector.
             # We can use the phase pattern as the direction.
             
             # Assuming attractor.phase_pattern is a vector of phases or directions.
             # In AttractorNetwork it is phases? No, "phase_pattern" in AttractorState.
             # Let's check AttractorState definition.
             pass
        
        # If we can't find an attractor, return None (or maybe a neutral phoneme 'a' vector?)
        # Let's return a neutral state based on phoneme 'a'
        try:
             neutral_vec = self.phoneme_mapper.map_phoneme('a')
             # Resize if dimensions differ (e.g. latent space dim vs phoneme dim)
             if neutral_vec.shape[0] != state.resonance_vector.shape[0]:
                 # Handle dimension mismatch if any. Assume they match for now as per system design (64 default?)
                 # Phoneme mapper default is 24. Latent space 64.
                 # We need to pad or project.
                 resized_vec = np.zeros(state.resonance_vector.shape[0])
                 d = min(len(neutral_vec), len(resized_vec))
                 resized_vec[:d] = neutral_vec[:d]
                 neutral_vec = resized_vec
                 
             return replace(state, resonance_vector=neutral_vec, stability_score=0.5, mu_value=0.1)
        except:
             return None

    def _project_to_phoneme_region(self, state: ResonanceState) -> ResonanceState:
        """
        Snap the drifted vector to the nearest valid phoneme vector.
        """
        current_vec = state.resonance_vector
        best_dist = float('inf')
        best_vec = current_vec
        
        # Check against all known phoneme vectors
        for phoneme, p_vec in self.phoneme_mapper.phoneme_vectors.items():
            # Pad p_vec to match current_vec dimension if needed
            if p_vec.shape[0] != current_vec.shape[0]:
                padded_p = np.zeros_like(current_vec)
                d = min(len(p_vec), len(padded_p))
                padded_p[:d] = p_vec[:d]
                target = padded_p
            else:
                target = p_vec
                
            dist = np.linalg.norm(current_vec - target)
            if dist < best_dist:
                best_dist = dist
                best_vec = target
                
        # We blend the current vector with the best phoneme vector to correct it, rather than hard replacing
        # This preserves some context while restoring structure.
        # factor alpha = 0.5
        corrected_vec = 0.5 * current_vec + 0.5 * best_vec
        
        return replace(state, resonance_vector=corrected_vec)
        
    def _log_error(self, error_type: str, message: str):
        self.error_log.append({
            "type": error_type,
            "message": message,
            "timestamp": np.datetime64('now')
        })


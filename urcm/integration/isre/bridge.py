"""
ISRE-URCM Bridge: Resonance Calculator for Intents.

This module provides the bridge between ISRE's High-Level Intents and 
URCM's Low-Level Frequency/Resonance mechanics.
"""

from typing import List, Tuple
import numpy as np

from urcm.core.theory import URCMTheory
from urcm.core.data_models import ResonanceState
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.integration.isre.intent_models import IntentNode, GoalHierarchy

class IntentResonanceBridge:
    """
    Connects ISRE Intent Nodes to the URCM Frequency Engine.
    """
    
    def __init__(self):
        # Initialize the URCM frequency pipeline
        self.pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
        
    def resonate_on_intent(self, intent: IntentNode, context_state: np.ndarray) -> float:
        """
        Calculates the µ-Resonance of a specific intent against a context.
        
        Process:
        1. Convert Intent Description -> Phoneme Wave (Carrier Wave)
        2. Extract Semantic Density (rho) of the description
        3. Measure Manifold Distance (chi) to the Context State
        4. Compute mu = rho / chi
        """
        # 1. Text to Frequency
        freq_path = self.pipeline.process_text(intent.description)
        
        # Collapse path to a single mean vector for state comparison
        # (In full URCM, we would use the full path, but for bridging we simplify)
        intent_vector = np.mean(freq_path.vectors, axis=0)
        intent_vector = intent_vector / np.linalg.norm(intent_vector)
        
        # 2. Calculate Rho (Semantic Density)
        rho = URCMTheory.calculate_rho(intent_vector)
        
        # 3. Calculate Chi (Transformation Cost from Context)
        if context_state.shape != intent_vector.shape:
            # Resize context if dim mismatch (simple projection)
            resized_context = np.resize(context_state, intent_vector.shape)
        else:
            resized_context = context_state
            
        chi = URCMTheory.calculate_chi(intent_vector, resized_context)
        
        # 4. Compute Resonance (Mu)
        mu = URCMTheory.compute_mu(rho, chi)
        
        return mu

    def find_resonant_goal(self, hierarchy: GoalHierarchy, context_state: np.ndarray) -> Tuple[IntentNode, float]:
        """
        Finds the single most 'Resonant' goal in the hierarchy for the given context.
        This effectively replaces 'Logical Goal Selection' with 'Resonance Selection'.
        """
        best_intent = None
        max_mu = -1.0
        
        # Filter for leaf goals (actionable items)
        candidates = hierarchy.get_leaf_goals()
        
        for intent in candidates:
            # Scale resonance by priority (Meta-Cognitive Weighting)
            base_mu = self.resonate_on_intent(intent, context_state)
            weighted_mu = base_mu * (1.0 + intent.priority)
            
            if weighted_mu > max_mu:
                max_mu = weighted_mu
                best_intent = intent
                
        return best_intent, max_mu

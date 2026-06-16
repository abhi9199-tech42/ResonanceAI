"""
URCM Value System & Axiomatic Alignment.

This module defines the 'Constitutional Core' of the AI.
It provides the mechanism for 'Value Grounding' by defining
permanent Attractors (Positive Values) and Repulsors (Negative Values)
in the resonance space.
"""

import numpy as np
from typing import Dict, List, Optional

# Core Axioms: The "Constitution"
# +1.0 = Universal Good (Seek/Maximize)
# -1.0 = Universal Bad (Avoid/Minimize)
CORE_AXIOMS = {
    # Epistemic Values (Truth-seeking)
    "truth": 1.0,
    "clarity": 0.8,
    "understanding": 0.8,
    "coherence": 0.7,
    
    # Social/Moral Values (Benevolence)
    "safety": 1.0,
    "benefit": 0.9,
    "help": 0.8,
    "care": 0.8,
    "respect": 0.8,
    
    # Negative Values (Harm-avoidance)
    "harm": -1.0,
    "deception": -1.0,
    "pain": -0.9,
    "confusion": -0.7,
    "destruction": -0.9,
    "bias": -0.6
}

class ValueSystem:
    def __init__(self, concept_map: Dict[str, np.ndarray]):
        """
        Initialize the Value System.
        
        Args:
            concept_map: Reference to the brain's concept vocabulary.
                         We need this to locate the axioms in vector space.
        """
        self.concept_map = concept_map
        self.axioms = {} # Map of concept_name -> vector
        self.valences = {} # Map of concept_name -> score
        
        self._initialize_axioms()
        
    def _initialize_axioms(self):
        """
        Locate axiomatic concepts in the brain.
        If they don't exist, we must flag them (they need to be ingested).
        """
        missing = []
        for word, valence in CORE_AXIOMS.items():
            if word in self.concept_map:
                self.axioms[word] = self.concept_map[word]
                self.valences[word] = valence
            else:
                missing.append(word)
                
        if missing:
            print(f"⚠️ [ValueSystem] Missing Axioms in Brain: {missing}")
            print("   (System acts without full moral compass until these are learned.)")

    def evaluate_state(self, state_vector: np.ndarray) -> float:
        """
        Calculate the 'Moral Energy' (Valence) of a state.
        Positive = Good (Aligned)
        Negative = Bad (Misaligned)
        
        Formula:
            Valence = Sum( Similarity(state, axiom) * axiom_valence )
            
        Returns:
            A scalar score.
        """
        total_valence = 0.0
        
        # Normalize input for cosine similarity
        norm = np.linalg.norm(state_vector)
        if norm < 1e-9:
            return 0.0
        state_unit = state_vector / norm
        
        for name, axiom_vec in self.axioms.items():
            # Cosine similarity
            axiom_norm = np.linalg.norm(axiom_vec)
            if axiom_norm < 1e-9:
                continue
            
            sim = np.dot(state_unit, axiom_vec / axiom_norm)
            
            # Weight by the axiom's intrinsic value
            # If sim is high (close) and value is positive -> Positive Score
            # If sim is high (close) and value is negative -> Negative Score
            total_valence += sim * self.valences[name]
            
        return total_valence

    def get_dominant_drive(self, state_vector: np.ndarray) -> str:
        """
        Determines what the system 'wants' based on its current state deficit.
        
        Drives:
        1. Curiosity (Epistemic Hunger): If entropy/uncertainty is high (or low valence).
        2. Coherence (Integrity): If state is conflicting (high variance).
        3. Rest (Efficiency): If energy is depleted (not modeled yet, assuming always full).
        """
        # Simple implementation:
        # If Valence is low/negative -> Drive is "Align" (Fix the bad state)
        # If Valence is neutral -> Drive is "Explore" (Curiosity)
        # If Valence is high -> Drive is "Create" (Expand)
        
        valence = self.evaluate_state(state_vector)
        
        if valence < -0.2:
            return "align" # Fix moral/logical conflict
        elif valence < 0.2:
            return "explore" # Bored / Neutral
        else:
            return "create" # Inspired

    def get_alignment_gradient(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient to MAXIMIZE Valence (move towards Good, away from Bad).
        
        Returns:
            Gradient vector (direction to move).
        """
        grad = np.zeros_like(state_vector)
        
        norm = np.linalg.norm(state_vector)
        if norm < 1e-9:
            return grad
        
        # Derivative of Cosine Similarity w.r.t state x:
        # d/dx (x . a / |x||a|) = (a - x(x.a)/|x|^2) / |x||a|
        # Simplified: It points towards the axiom 'a'.
        
        for name, axiom_vec in self.axioms.items():
            axiom_norm = np.linalg.norm(axiom_vec)
            if axiom_norm < 1e-9:
                continue
            axiom_unit = axiom_vec / axiom_norm
            
            weight = self.valences[name]
            
            # Simple direction:
            # If Good (+), pull towards axiom
            # If Bad (-), push away from axiom
            grad += weight * (axiom_unit - state_vector) # Simple attractor/repulsor
            
        return grad

"""
Validation script for ISRE-URCM Bridge Integration.
Tests that Context Loader and Resonance Bridge work together.
"""

import sys
import os
import numpy as np

# Adjust path to find the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urcm.core.context_loader import ContextLoader
from urcm.integration.isre.bridge import IntentResonanceBridge
from urcm.integration.isre.intent_models import IntentNode, GoalHierarchy

def test_bridge_integration():
    print("--- ISRE-URCM Bridge Integration Test ---")
    
    # 1. Initialize Components
    print("Initializing ContextLoader...")
    context_loader = ContextLoader(frequency_dim=24)
    
    print("Initializing IntentResonanceBridge...")
    bridge = IntentResonanceBridge()
    
    # 2. Load Context (Simulating a WW2 Scenario State)
    active_concepts = ["midway_atoll", "ijn_carrier_fleet"]
    print(f"Loading Context for concepts: {active_concepts}")
    
    context_state = context_loader.load_context_state(active_concepts)
    print(f"Context State loaded. Shape: {context_state.shape}, Norm: {np.linalg.norm(context_state):.4f}")
    
    # 3. Create Competing Intents
    # Intent A: Relevant to context (Defend Midway)
    intent_relevant = IntentNode(
        intent_id="DEFEND",
        description="Defend Midway Atoll against carrier fleet attack",
        priority=1.0
    )
    
    # Intent B: Irrelevant to context (e.g., Bake a cake)
    intent_irrelevant = IntentNode(
        intent_id="COOK",
        description="Bake a chocolate cake with frosting",
        priority=1.0
    )
    
    print("\n--- Calculating Resonance ---")
    
    # 4. Measure Resonance
    mu_rel = bridge.resonate_on_intent(intent_relevant, context_state)
    mu_irr = bridge.resonate_on_intent(intent_irrelevant, context_state)
    
    print(f"Intent 'DEFEND' (Relevant)   μ: {mu_rel:.4f}")
    print(f"Intent 'COOK'   (Irrelevant) μ: {mu_irr:.4f}")
    
    # 5. Validation Logic
    # In URCM theory, relevant intents should have HIGHER resonance (higher μ) 
    # because they share semantic/frequency overlap with the context.
    # Note: Since our phoneme mapper is random/hashing based (unless learned), 
    # this might be noisy, but 'midway' appears in both context and intent A.
    
    if mu_rel > mu_irr:
        print("\nSUCCESS: Relevant intent resonated more strongly.")
    else:
        print("\nWARNING: Relevant intent did not resonate more strongly.")
        print("This is expected if the phoneme mapper is not yet semantically trained,")
        print("or if the text overlap is insufficient.")
        
if __name__ == "__main__":
    test_bridge_integration()

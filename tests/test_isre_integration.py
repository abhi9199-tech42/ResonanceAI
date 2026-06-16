"""
Integration Test: Can URCM pick the right WW2 strategy using Resonance?
"""

import json
import numpy as np
import os
import sys

# Ensure project root is in path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urcm.integration.isre.intent_models import IntentNode, GoalHierarchy
from urcm.integration.isre.bridge import IntentResonanceBridge

def load_scenarios():
    path = os.path.join(os.path.dirname(__file__), "data", "isre_scenarios", "ww2_scenarios.json")
    with open(path, 'r') as f:
        return json.load(f)

def test_ww2_resonance_selection():
    """
    Test if the Resonance Engine selects the semantically correct strategy
    given a specific context vector.
    """
    scenarios = load_scenarios()
    midway = scenarios[0] # WW2_PACIFIC_1942
    
    # 1. Build the Goal Hierarchy
    hierarchy = GoalHierarchy(root_id="ROOT")
    for item in midway['intents']:
        node = IntentNode(
            intent_id=item['id'],
            description=item['description'],
            priority=item['priority']
        )
        hierarchy.add_node(node)
        
    # 2. Define a Context via ContextLoader (The "World State")
    from urcm.core.context_loader import ContextLoader
    context_loader = ContextLoader()
    
    # We load the context state based on what's active in the 'World'
    # The KB knows that 'ijn_carrier_fleet' implies [Akagi, Kaga, threat=critical]
    active_concepts = ["ijn_carrier_fleet"]
    context_vector = context_loader.load_context_state(active_concepts)
    
    context_description = f"Active Concepts: {active_concepts}"
    
    # 3. Find the Resonant Goal
    bridge = IntentResonanceBridge()
    print(f"Context: {context_description}")
    print("-" * 40)
    
    best_intent, max_mu = bridge.find_resonant_goal(hierarchy, context_vector)
    
    print(f"Winner: {best_intent.intent_id}")
    print(f"Score (mu): {max_mu:.4f}")
    print(f"Description: {best_intent.description}")
    
    # 4. Success Criteria
    # The context is explicitly about striking carriers. 
    # The bridge should prefer AMBUSH_CARRIERS > DEFEND_MIDWAY > PRESERVE_FLEET
    
    if best_intent.intent_id == "AMBUSH_CARRIERS":
        print("\nSUCCESS: System autonomously selected the correct tactical response based on semantic resonance!")
    else:
        print(f"\nFAILURE: System selected {best_intent.intent_id} instead of AMBUSH_CARRIERS.")

if __name__ == "__main__":
    test_ww2_resonance_selection()

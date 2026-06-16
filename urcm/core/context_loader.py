"""
URCM Context Loader & Knowledge Interface

This module implements the Context/Knowledge interface that was originally part of ISRE.
It serves two purposes:
1. Provide access to valid knowledge/facts (the "World Structure").
2. Convert that knowledge into a Resonance Field (Context State) for the reasoning engine.

Ported from ISRE/knowledge/engine.py
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

# We'll need the pipeline to sonify context
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.data_models import ResonanceState

@dataclass
class KnowledgeQueryResult:
    """
    Standardized result from any knowledge source.
    """
    source_id: str
    fact_id: str
    content: Any
    confidence: float
    metadata: Dict[str, Any]

class KnowledgeQueryEngine:
    """
    Interfaces with external structured knowledge sources.
    Adapts ISRE's knowledge structure to URCM.
    """
    
    def __init__(self):
        # In a real system, this would connect to databases/APIs
        # This is a local mock based on the ISRE implementation
        self._knowledge_base = {
            # Basic Concepts
            "apple": {"category": "fruit", "edible": True, "color": ["red", "green"]},
            "run": {"category": "action", "energy_cost": "high"},
            "physics_gravity": {"value": 9.81, "unit": "m/s^2"},
            
            # WW2 Concepts (for scenario testing)
            "midway_atoll": {
                "type": "location",
                "strategic_value": "high", 
                "coordinates": "28.2N 177.3W",
                "controlled_by": "USA"
            },
            "ijn_carrier_fleet": {
                "type": "military_unit",
                "composition": ["Akagi", "Kaga", "Soryu", "Hiryu"],
                "threat_level": "critical",
                "last_known_loc": "unknown"
            },
            "arnhem_bridge": {
                "type": "infrastructure",
                "value": "strategic_crossing",
                "status": "intact",
                "defended_by": "German Panzer Division"
            }
        }
        self._cache: Dict[str, KnowledgeQueryResult] = {}

    def query(self, concept_key: str) -> Optional[KnowledgeQueryResult]:
        """
        Retrieves knowledge for a specific concept.
        Returns None if knowledge is missing.
        """
        concept_key = concept_key.lower()
        
        if concept_key in self._cache:
            return self._cache[concept_key]

        data = self._knowledge_base.get(concept_key)
        if data:
            res = KnowledgeQueryResult(
                source_id="internal_kb_mock",
                fact_id=f"fact_{concept_key}",
                content=data,
                confidence=1.0,
                metadata={"schema": "1.0"}
            )
            self._cache[concept_key] = res
            return res
        return None

    def update_knowledge(self, concept_key: str, data: Any):
        """Update or add a new fact to the knowledge base."""
        self._knowledge_base[concept_key.lower()] = data
        if concept_key.lower() in self._cache:
            del self._cache[concept_key.lower()]

class ContextLoader:
    """
    Loads knowledge and transforms it into a URCM Context State (Resonance Field).
    This validates Requirement 1.5 (Context Integration).
    """
    
    def __init__(self, frequency_dim: int = 24):
        self.kb_engine = KnowledgeQueryEngine()
        self.pipeline = PhonemeFrequencyPipeline(frequency_dim=frequency_dim)
        self.frequency_dim = frequency_dim
        
    def load_context_state(self, active_concepts: List[str]) -> np.ndarray:
        """
        Converts a list of active concepts into a unified Context Resonance Field.
        
        Process:
        1. Query KB for each concept.
        2. Convert valid facts to text descriptions.
        3. Sonify descriptions into frequency vectors.
        4. Superimpose vectors to create the Context Field.
        """
        vectors = []
        
        for concept in active_concepts:
            result = self.kb_engine.query(concept)
            if result:
                # Convert structured content to comparable string
                # e.g., "midway_atoll type location strategic_value high"
                content_str = self._stringify_content(concept, result.content)
                
                # Sonify
                path = self.pipeline.process_text(content_str)
                
                # Mean pool to get stable representation of the concept
                concept_vector = np.mean(path.vectors, axis=0)
                
                # Normalize
                norm = np.linalg.norm(concept_vector)
                if norm > 1e-6:
                    concept_vector = concept_vector / norm
                    
                vectors.append(concept_vector)
                
        if not vectors:
            # If no context, return neutral state (zeros)
            return np.zeros(self.frequency_dim)
            
        # Superposition of all context vectors
        # In simple resonance, this is vector addition/averaging
        context_field = np.mean(vectors, axis=0)
        
        # Normalize final field
        final_norm = np.linalg.norm(context_field)
        if final_norm > 1e-6:
            context_field = context_field / final_norm
            
        return context_field
        
    def _stringify_content(self, concept: str, content: Any) -> str:
        """Helper to convert dictionary/any content to sonifiable string."""
        if isinstance(content, dict):
            # "concept key value key value"
            tokens = [concept]
            for k, v in content.items():
                tokens.append(str(k))
                if isinstance(v, list):
                    tokens.extend([str(i) for i in v])
                else:
                    tokens.append(str(v))
            return " ".join(tokens)
        return f"{concept} {str(content)}"

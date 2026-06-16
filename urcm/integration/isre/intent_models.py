"""
ISRE Intent Data Models (Reconstructed for URCM Integration)

This module implements the core Intent structures required to interface with
URCM's resonance engine. It mirrors the 'IntentNode' and 'GoalHierarchy'
structures typically found in the ISRE system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

@dataclass
class IntentNode:
    """
    Represents a single semantic intent or goal unit from the ISRE system.
    """
    intent_id: str
    description: str
    priority: float  # 0.0 to 1.0
    constraints: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Metadata for resonance mapping
    keywords: List[str] = field(default_factory=list)
    preconditions: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.description:
            raise ValueError("Intent description cannot be empty")
        if not (0.0 <= self.priority <= 1.0):
            raise ValueError("Priority must be between 0.0 and 1.0")

@dataclass
class GoalHierarchy:
    """
    Represents a full hierarchy of intents (Goal Graph).
    """
    root_id: str
    nodes: Dict[str, IntentNode] = field(default_factory=dict)
    
    def add_node(self, node: IntentNode):
        self.nodes[node.intent_id] = node
        
    def get_children(self, node_id: str) -> List[IntentNode]:
        if node_id not in self.nodes:
            return []
        return [self.nodes[cid] for cid in self.nodes[node_id].children_ids]

    def get_leaf_goals(self) -> List[IntentNode]:
        """Returns all nodes that have no children."""
        return [n for n in self.nodes.values() if not n.children_ids]

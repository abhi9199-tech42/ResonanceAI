import time
import numpy as np
from typing import List, Optional, Dict, Any
import uuid

class Intent:
    """
    Represents a discrete Unit of Intent (Goal/Task) in Working Memory.
    """
    def __init__(self, 
                 description: str, 
                 target_concept_name: Optional[str] = None,
                 target_vector: Optional[np.ndarray] = None,
                 constraints: Optional[List[Any]] = None,
                 logic_gates: Optional[List[Dict]] = None,
                 priority: float = 1.0,
                 timeout_steps: int = 20):
        
        self.id = str(uuid.uuid4())[:8]
        self.description = description
        self.target_concept_name = target_concept_name
        self.target_vector = target_vector
        self.constraints = constraints if constraints is not None else []
        self.logic_gates = logic_gates if logic_gates is not None else []
        self.priority = priority
        self.created_at = time.time()
        self.steps_taken = 0
        self.timeout_steps = timeout_steps
        self.status = "active" # active, completed, failed, suspended

    def __repr__(self):
        return f"<Intent '{self.description}' ({self.status})>"

class WorkingMemory:
    """
    The Executive Workspace (Left Brain Core).
    Holds the 'Stack' of active intentions and manages context switching.
    """
    def __init__(self):
        self.intent_stack: List[Intent] = []
        self.completed_log: List[Intent] = []
        
    def add_intent(self, intent: Intent):
        """Pushes a new intent onto the stack (Focus Shift)."""
        # If there's an active intent, suspend it? 
        # For now, just simple stack behavior. Top is active.
        self.intent_stack.append(intent)
        print(f"[WM] ➕ Added Intent: {intent.description}")
        
    def pop_intent(self) -> Optional[Intent]:
        """Removes the current intent (Completion/Failure)."""
        if self.intent_stack:
            intent = self.intent_stack.pop()
            print(f"[WM] ➖ Popped Intent: {intent.description}")
            return intent
        return None
        
    def get_current_intent(self) -> Optional[Intent]:
        """Peeks at the active intent."""
        if self.intent_stack:
            return self.intent_stack[-1]
        return None
        
    def complete_intent(self, intent: Intent, success: bool = True):
        """Marks intent as complete/failed and moves to log."""
        intent.status = "completed" if success else "failed"
        self.completed_log.append(intent)
        # Ensure it's removed from stack if it was top
        if self.intent_stack and self.intent_stack[-1].id == intent.id:
            self.pop_intent()
            
    def clear(self):
        self.intent_stack = []
        self.completed_log = []

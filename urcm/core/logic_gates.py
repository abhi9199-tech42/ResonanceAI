from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class GeometricLogic:
    """
    Implements Logic Gates using Energy Landscapes (Left Brain).

    Instead of binary True/False, we use 'Energy Modifiers':
    - AND(A, B): Creates a basin where BOTH A and B are active.
    - OR(A, B): Creates a double-well basin (bistable).
    - NOT(A): Creates a hill (repeller) at A.
    - IMPLIES(A, B): Creates a directional gradient flow from A to B.
    - PONENS(A, B): Alias of IMPLIES (Modus Ponens).
    - TOLLENS(A, B): If NOT(B), then NOT(A) (repel from A when far from B).
    - TRANSITIVE(A, B, C): If near A→B and B→C, pull towards C.
    - SYLLOGISM(S, M, P): If S is M and all M are P, pull S towards P.
    """

    def __init__(self, concept_map: Dict[str, np.ndarray]):
        self.concept_map = concept_map

    def get_vector(self, concept: str) -> Optional[np.ndarray]:
        return self.concept_map.get(concept)

    def apply_constraint(self,
                       current_state: np.ndarray,
                       logic_type: str,
                       operands: List[str],
                       weight: float = 1.0) -> np.ndarray:
        """
        Calculates the ENERGY GRADIENT for a logical constraint.
        Returns a vector (direction) to move the state towards satisfaction.
        """

        vectors = [self.get_vector(op) for op in operands]
        if any(v is None for v in vectors):
            return np.zeros_like(current_state) # Cannot apply if concepts unknown

        grad = np.zeros_like(current_state)
        cu = current_state / (np.linalg.norm(current_state) + 1e-9)
        units = [v / (np.linalg.norm(v) + 1e-9) for v in vectors]

        if logic_type == "NOT":
            a = units[0]
            grad = (cu - a) * weight

        elif logic_type == "AND":
            a, b = units[0], units[1]
            target = a + b
            target = target / (np.linalg.norm(target) + 1e-9)
            grad = (target - cu) * weight

        elif logic_type == "OR":
            a, b = units[0], units[1]
            sim_a = float(np.dot(cu, a))
            sim_b = float(np.dot(cu, b))
            target = a if sim_a >= sim_b else b
            grad = (target - cu) * weight

        elif logic_type == "IMPLIES":
            a, b = units[0], units[1]
            sim_a = float(np.dot(cu, a))
            if sim_a > 0.6:
                grad = (b - cu) * weight * (sim_a - 0.6)

        elif logic_type == "PONENS":
            a, b = units[0], units[1]
            sim_a = float(np.dot(cu, a))
            if sim_a > 0.6:
                grad = (b - cu) * weight * (sim_a - 0.6)

        elif logic_type == "TOLLENS":
            a, b = units[0], units[1]
            sim_b = float(np.dot(cu, b))
            if sim_b < 0.2:
                repel = (cu - a)
                grad = repel * weight * (0.2 - sim_b)

        elif logic_type == "TRANSITIVE":
            if len(units) >= 3:
                a, b, c = units[0], units[1], units[2]
                sa = float(np.dot(cu, a))
                sb = float(np.dot(cu, b))
                if sa > 0.6 or sb > 0.6:
                    closeness = max(0.0, max(sa - 0.6, sb - 0.6))
                    grad = (c - cu) * weight * max(0.2, closeness)

        elif logic_type == "SYLLOGISM":
            if len(units) >= 3:
                s, m, p = units[0], units[1], units[2]
                ss = float(np.dot(cu, s))
                sm = float(np.dot(cu, m))
                if ss > 0.5 or sm > 0.6:
                    closeness = max(0.0, max(ss - 0.5, sm - 0.6))
                    grad = (p - cu) * weight * max(0.2, closeness)

        return grad

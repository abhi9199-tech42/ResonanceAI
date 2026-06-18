import re
from typing import Dict, List, Optional

import numpy as np

from urcm.core.long_term_memory import LongTermMemory
from urcm.core.reasoning import ReasoningEngine
from urcm.core.sanskrit_bridge import SanskritBridge
from urcm.core.sanskrit_grammar import SanskritGrammar
from urcm.core.symbolic_engine import SymbolicEngine
from urcm.core.working_memory import Intent, WorkingMemory


class ExecutiveController:
    """
    The 'Left Brain' Executive.
    Orchestrates the Right Brain (ReasoningEngine) using Working Memory (Intent Stack).
    Now augmented with Long Term Memory (Library) and Symbolic Engine (Calculator).
    """
    def __init__(self, brain_path: str = "urcm_identity.pkl"):
        self.engine = ReasoningEngine(brain_path)
        self.memory = WorkingMemory()

        # AGI Components
        self.long_term_memory = LongTermMemory(vector_dim=self.engine.l2_dim)
        self.symbolic_engine = SymbolicEngine()
        self.grammar = SanskritGrammar()
        self.bridge = SanskritBridge()

        self.current_state = None

    def set_initial_state(self, concept_text: str):
        """Sets the starting thought of the Right Brain."""
        vec = self.engine.get_concept_vector(concept_text)
        if vec is not None:
            self.current_state = vec
            print(f"[Exec] Initialized state to: {concept_text}")
        else:
            print(f"[Exec] ⚠️ Unknown concept: '{concept_text}'. Using random initialization.")
            # Fallback: Initialize with random vector
            self.current_state = np.random.randn(self.engine.l2_dim)
            # Normalize
            self.current_state = self.current_state / np.linalg.norm(self.current_state)

    def reload_brain(self):
        """Reloads the ReasoningEngine to pick up new knowledge."""
        print("[Exec] 🔄 Reloading Brain (Plasticity Update)...")
        # Preserve state if possible
        old_state = self.current_state.copy() if self.current_state is not None else None

        self.engine = ReasoningEngine(self.engine.brain_data_path if hasattr(self.engine, 'brain_data_path') else "urcm_identity.pkl")

        # Restore state (if dimensions match)
        if old_state is not None:
             # Check if dimension changed (e.g. if we resized W_res)
             if old_state.shape[0] == self.engine.l2_dim:
                 self.current_state = old_state
             else:
                 print("[Exec] ⚠️ State dimension mismatch after reload. Resetting to neutral.")
                 self.current_state = np.zeros(self.engine.l2_dim)

    def store_thought(self, text: str, vector: np.ndarray = None, tags: List[str] = []):
        """Saves a thought/concept to the Long Term Memory (Library)."""
        if vector is None:
            vector = self.current_state
        if vector is None:
            print("[Exec] ⚠️ Cannot store thought: No vector provided.")
            return

        self.long_term_memory.add(text, vector, tags, source="executive")

    def recall_thought(self, query: str = None, vector: np.ndarray = None, k: int = 3):
        """Retrieves related thoughts from Long Term Memory."""
        if vector is None:
            if query:
                vector = self.engine.get_concept_vector(query)
            else:
                vector = self.current_state

        if vector is None:
            print("[Exec] ⚠️ Cannot recall: No query vector available.")
            return []

        return self.long_term_memory.retrieve(vector, k=k)

    def calculate(self, expression: str, as_script: bool = False):
        """Offloads a calculation to the Symbolic Engine."""
        print(f"[Exec] 🧮 Calculating ({'script' if as_script else 'expr'}): {expression[:50]}...")
        if as_script:
            success, result, error = self.symbolic_engine.execute_script(expression)
        else:
            success, result, error = self.symbolic_engine.evaluate(expression)

        if success:
            print(f"[Exec] ✅ Result: {str(result)[:100]}")
            return result
        else:
            print(f"[Exec] ❌ Calculation Error: {error}")
            return None

    def add_goal(self, description: str, target_concept: str = None, priority: float = 1.0):
        """Adds a high-level goal to Working Memory."""
        target_vec = None
        if target_concept:
            target_vec = self.engine.get_concept_vector(target_concept)

        intent = Intent(
            description=description,
            target_concept_name=target_concept,
            target_vector=target_vec,
            priority=priority
        )
        self.memory.add_intent(intent)
        print(f"[WM] ➕ Added Intent: {description}")

    def learn_logic(self, premise: str, conclusion: str):
        """
        Teaches the system a logical transition (Premise -> Conclusion).
        Wraps ReasoningEngine's learn_transition.
        Example: "Fire" -> "Hot"
        """
        vec_p = self.engine.get_concept_vector(premise)
        vec_c = self.engine.get_concept_vector(conclusion)

        if vec_p is None:
            print(f"[Exec] 🐣 Registering new concept: '{premise}'")
            self.engine.add_concept_from_examples(premise, [])
            vec_p = self.engine.get_concept_vector(premise)

        if vec_c is None:
            print(f"[Exec] 🐣 Registering new concept: '{conclusion}'")
            self.engine.add_concept_from_examples(conclusion, [])
            vec_c = self.engine.get_concept_vector(conclusion)

        print(f"[Exec] 🎓 Learning: {premise} -> {conclusion}")
        # Call the reasoning engine's learning method
        # Assuming learn_transition takes (state_prev, state_next)
        self.engine.learn_transition(vec_p, vec_c, learning_rate=0.2) # Lower LR for stability

        # Save the brain immediately to persist logic
        self.engine.save_brain()

    def run_loop(self, max_steps: int = 20):
        """
        Runs the Unified Cognitive Loop.
        Integrates Perception (LTM), Reasoning (Net), Logic (Symbolic), and Expression (Grammar).
        """
        if self.current_state is None:
            # Check LTM for last state, or random
            print("[Exec] ⚠️ No active state. Initializing random thought.")
            self.current_state = np.random.randn(self.engine.l2_dim)
            self.current_state /= np.linalg.norm(self.current_state)

        trajectory = []
        state_trajectory = [self.current_state.copy()]

        # 0. Intrinsic Motivation Check (Self-Directed Goal Formation)
        active_intent = self.memory.get_current_intent()
        if not active_intent:
            print("\n[Exec] 💤 Mind is idle. Checking drives...")
            drive = self.engine.values.get_dominant_drive(self.current_state)
            print(f"[Exec] 🔋 Dominant Drive: {drive.upper()}")

            if drive == "explore":
                # Create a curiosity goal
                print("[Exec] 💡 Creating Intrinsic Goal: Explore new concepts")
                self.add_goal("explore_novelty", priority=0.5)
            elif drive == "align":
                # Create a coherence goal
                print("[Exec] 🔧 Creating Intrinsic Goal: Reduce Cognitive Dissonance")
                # Target 'coherence' concept if available
                if "coherence" in self.engine.concept_map:
                    self.add_goal("seek_coherence", target_concept="coherence", priority=0.8)
                else:
                    self.add_goal("stabilize_thought", priority=0.8)

            # Refresh intent
            active_intent = self.memory.get_current_intent()

        # 1. Perception / Context Retrieval (The "Look Before You Leap" Step)
        if active_intent:
            print(f"\n[Exec] 🎯 Focus: {active_intent.description}")
            # Check LTM for relevant past experiences
            context = self.recall_thought(query=active_intent.description, k=1)
            if context:
                print(f"[Exec] 📚 Recalled context: '{context[0][0]['text']}'")
                # TODO: Ideally, blend this context vector into current_state

        for t in range(max_steps):
            active_intent = self.memory.get_current_intent()

            # 2. Configure Attention (Constraints from Intent)
            constraints = []
            goal_vec = None
            logic_gates = []

            if active_intent:
                if active_intent.target_vector is not None:
                    goal_vec = active_intent.target_vector
                    pr = active_intent.priority if isinstance(active_intent.priority, (int, float)) else 1.0
                    constraints.append((active_intent.target_vector, 5.0 * pr))
                logic_gates.extend(active_intent.logic_gates)

            # 3. Execute Step (Right Brain Associative Leap)
            descent_steps = 10 if active_intent else 3

            next_state, word, signals = self.engine.step(
                self.current_state,
                goal_vec,
                constraints,
                logic_gates,
                descent_steps=descent_steps
            )

            self.current_state = next_state
            state_trajectory.append(next_state.copy())

            # 4. Neuro-Symbolic Intercept (The "Sanity Check")
            # If the thought stream implies a calculation, pause and use the Symbolic Engine.
            # Simple heuristic: if word contains digits or math symbols
            if re.search(r'\d', word) or word in ['+', '-', '*', '/', '=']:
                # Look back in trajectory to form an expression
                # (Very basic sliding window for now)
                context_window = trajectory[-2:] + [word]
                expr_candidate = " ".join(context_window)
                # Heuristic: try to eval if it looks like math
                if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', expr_candidate):
                     res = self.calculate(expr_candidate)
                     if res is not None:
                         print(f"[Exec] 🧠 Integrating Logic Result: {res}")
                         # Ideally: Update current_state with vector for the result
                         # For now, just append to trajectory
                         word = f"{word} (={res})"

            # Debug distance
            dist_str = ""
            if active_intent and active_intent.target_vector is not None:
                dist = np.linalg.norm(self.current_state - active_intent.target_vector)
                dist_str = f" | DistToGoal={dist:.4f}"

            print(f"[Exec] Step {t}: Thought='{word}' | Focus={signals['focus']:.2f}{dist_str}")

            trajectory.append(word)

            # 5. Check Completion
            if active_intent and active_intent.target_vector is not None:
                dist = np.linalg.norm(self.current_state - active_intent.target_vector)
                if word == active_intent.target_concept_name or dist < 0.5:
                    print(f"[Exec] ✅ Goal '{active_intent.description}' Complete!")
                    self.memory.complete_intent(active_intent, success=True)

                    # Store success in LTM
                    final_thought = f"Achieved {active_intent.description} via {', '.join(trajectory[-5:])}"
                    self.store_thought(final_thought, self.current_state, tags=["success", "goal"])

                    # --- LEARNING STEP (Plasticity) ---
                    # We reached the goal. Reinforce the path that got us here.
                    print("[Exec] 🧠 Consolidating Knowledge (Hebbian Update)...")
                    # Learn the last N steps
                    steps_to_learn = min(len(state_trajectory)-1, 5)
                    for i in range(len(state_trajectory)-steps_to_learn-1, len(state_trajectory)-1):
                        s_prev = state_trajectory[i]
                        s_next = state_trajectory[i+1]
                        self.engine.learn_transition(s_prev, s_next, learning_rate=0.05)

                    self.engine.save_brain()

                    break

            # 6. Handle Frustration (Switching/Giving Up)
            if signals["frustration"] > 0.8:
                 print("[Exec] ⚠️ High Frustration. Re-evaluating strategy...")
                 # In future: Pop intent, or add "Relax" sub-goal

            if not self.memory.get_current_intent():
                print("[Exec] All goals complete. Idle.")
                break

        # 7. Structure & Expression (The "Speech" Step)
        # Use Sanskrit Grammar to structure the raw stream
        print("\n[Exec] 🗣️  Structuring Thought Stream...")
        sanskrit_traj = self.bridge.translate_trajectory(trajectory)
        structured = self.grammar.structure_thought(sanskrit_traj)
        print(f"[Exec] 🕉️  Statement: {structured}")

        return trajectory

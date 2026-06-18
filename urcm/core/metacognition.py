"""
Metacognition Module: The Left Brain Monitor.

This module implements the "Observer" that watches the Right Brain's stream of consciousness.
It detects:
1. High Entropy (Confusion)
2. Loops (Stagnation)
3. Goal Drift (Distraction)

And triggers control signals (Gain, Noise, Focus) to correct the trajectory.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class MetacognitiveMonitor:
    def __init__(self, history_window: int = 5):
        self.history_window = history_window
        self.energy_history: List[float] = []
        self.state_history: List[np.ndarray] = []
        self.trajectory_history: List[str] = []

    def monitor(self,
                current_state: np.ndarray,
                current_energy: float,
                current_word: str) -> Dict[str, float]:
        """
        Analyzes the current thought state and returns control signals.

        Returns:
            Dict with keys:
            - 'frustration': 0.0 to 1.0 (Need to inject noise?)
            - 'focus': 0.0 to 1.0 (Need to increase gain?)
            - 'status': "stable", "looping", "confused", "drifting"
        """
        # Update History
        self.energy_history.append(current_energy)
        self.state_history.append(current_state.copy())
        self.trajectory_history.append(current_word)

        # Trim history
        if len(self.energy_history) > self.history_window:
            self.energy_history.pop(0)
            self.state_history.pop(0)
            self.trajectory_history.pop(0)

        # 1. Detect Loops (Repetition)
        loop_score = self._detect_loops()

        # 2. Detect Confusion (High Entropy/Energy)
        confusion_score = self._detect_confusion()

        # 3. Determine Control Signals
        signals = {
            "frustration": 0.0,
            "focus": 0.0,
            "status": "stable"
        }

        if loop_score > 0.6:
            signals["status"] = "looping"
            signals["frustration"] = loop_score # Inject noise to break loop

        elif confusion_score > 0.7:
            signals["status"] = "confused"
            signals["focus"] = confusion_score # Increase gain to sharpen

        return signals

    def _detect_loops(self) -> float:
        """
        Checks if the current word has appeared recently.
        Returns 0.0 (Unique) to 1.0 (Stuck).
        """
        if len(self.trajectory_history) < 2:
            return 0.0

        current = self.trajectory_history[-1]

        # Check simple repetition
        count = self.trajectory_history.count(current)
        if count > 1:
            # If repeated > 50% of window, high loop score
            return min(1.0, count / len(self.trajectory_history))

        return 0.0

    def _detect_confusion(self) -> float:
        """
        Checks if energy is consistently high.
        """
        if not self.energy_history:
            return 0.0

        avg_energy = sum(self.energy_history) / len(self.energy_history)

        # Normalize (Assuming max energy approx 1.0-2.0)
        return min(1.0, avg_energy)

    def check_goal_adherence(self, current_state: np.ndarray, goal_state: Optional[np.ndarray]) -> float:
        """
        Working Memory Check: Is the current thought getting closer to the goal?
        Returns Cosine Similarity (1.0 = Aligned, -1.0 = Opposed).
        """
        if goal_state is None:
            return 1.0 # No goal, so technically adhering to 'freedom'

        # Cosine Similarity
        norm_c = np.linalg.norm(current_state)
        norm_g = np.linalg.norm(goal_state)

        if norm_c == 0 or norm_g == 0:
            return 0.0

        return np.dot(current_state, goal_state) / (norm_c * norm_g)

    def get_control_signals(self,
                          current_state: np.ndarray,
                          current_energy: float,
                          current_word: str,
                          goal_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Unified Control Logic.
        """
        # Monitor Basics
        monitor_signals = self.monitor(current_state, current_energy, current_word)

        # Check Goal
        goal_score = self.check_goal_adherence(current_state, goal_state)

        # If drifting from goal (score < 0.2), increase focus
        if goal_state is not None and goal_score < 0.2:
            monitor_signals["status"] = "drifting"
            monitor_signals["focus"] = max(monitor_signals["focus"], 0.5) # Force focus
            # Add a 'steering' signal? For now, just High Focus.

        return monitor_signals

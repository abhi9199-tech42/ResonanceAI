"""
Hierarchical Resonance Encoder.

This module implements the "Phase 3" requirement of URCM:
Hierarchical Composition of Resonance States.

Layer 1 (Phonemes): Maps Phoneme Vectors (T x 24) -> Phoneme Resonance State (64).
Layer 2 (Concepts): Maps Sequence of L1 States (T x 64) -> Concept Resonance State (128).

This allows the system to understand "Words" as trajectories of "Phonemes",
and eventually "Sentences" as trajectories of "Words".
"""

from typing import Dict, List, Union, Tuple
import numpy as np
import os
import pickle

from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.data_models import FrequencyPath

class HierarchicalEncoder:
    """
    A multi-layer resonance encoder that builds higher-order concepts
    from sequences of lower-order resonance states.
    """
    
    def __init__(
        self,
        l1_input_dim: int = 24,
        l1_res_dim: int = 64,
        l2_res_dim: int = 1024
    ):
        """
        Initialize the hierarchical system.
        
        Args:
            l1_input_dim: Dimension of raw phoneme vectors (default 24).
            l1_res_dim: Dimension of Layer 1 resonance state (default 64).
            l2_res_dim: Dimension of Layer 2 resonance state (default 128).
        """
        self.l1_input_dim = l1_input_dim
        self.l1_res_dim = l1_res_dim
        self.l2_res_dim = l2_res_dim
        
        # Initialize Layer 1 (Phoneme -> Resonance)
        self.layer1 = ResonancePathEncoder(
            input_dim=l1_input_dim,
            resonance_dim=l1_res_dim,
            encoder_type="recurrent_numpy"
        )
        
        # Initialize Layer 2 (L1 Sequence -> Concept)
        # Note: Input to L2 is the Output of L1 (l1_res_dim)
        self.layer2 = ResonancePathEncoder(
            input_dim=l1_res_dim,
            resonance_dim=l2_res_dim,
            encoder_type="recurrent_numpy"
        )
        
        # Flag to track if L2 is trained
        self.l2_is_trained = False
        
    def encode_concept(self, phoneme_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encodes a sequence of phonemes into a high-level concept vector.
        
        Args:
            phoneme_vectors: Array of shape (T, l1_input_dim).
            
        Returns:
            Tuple[l2_final_state, l1_trajectory]:
                - l2_final_state: The high-level concept vector (l2_res_dim).
                - l1_trajectory: The sequence of L1 states used as input to L2 (T, l1_res_dim).
        """
        # 1. Run Layer 1 to get the trajectory of resonance states
        #    (The "Feeling" of the phonemes changing over time)
        #    Shape: (T, l1_res_dim)
        l1_trajectory = self.layer1.get_state_trajectory(phoneme_vectors)
        
        # 2. Run Layer 2 on the L1 trajectory
        #    (The "Meaning" emerging from the changing feelings)
        #    Shape: (l2_res_dim,)
        l2_final_state = self.layer2.encode_path(l1_trajectory)
        
        return l2_final_state, l1_trajectory

    def encode_concept_batch(self, phoneme_vectors_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batched version of encode_concept.
        
        Args:
            phoneme_vectors_batch: Array of shape (Batch, T, l1_input_dim).
            
        Returns:
            Tuple[l2_final_states, l1_trajectories]:
                - l2_final_states: (Batch, l2_res_dim)
                - l1_trajectories: (Batch, T, l1_res_dim)
        """
        # 1. Run Layer 1 Batch
        # Shape: (Batch, T, l1_res_dim)
        l1_trajectories = self.layer1.get_state_trajectory_batch(phoneme_vectors_batch)
        
        # 2. Run Layer 2 Batch on L1 trajectories
        # Shape: (Batch, l2_res_dim)
        l2_final_states = self.layer2.encode_path_batch(l1_trajectories)
        
        return l2_final_states, l1_trajectories

    def save_hierarchy(self, path: str = "urcm_hierarchy.pkl"):
        """Saves the weights of both layers."""
        data = {
            "layer1": {
                "W_in": self.layer1.W_in,
                "W_res": self.layer1.W_res,
                "W_out": self.layer1.W_out,
                "bias": self.layer1.bias
            },
            "layer2": {
                "W_in": self.layer2.W_in,
                "W_res": self.layer2.W_res,
                "W_out": self.layer2.W_out,
                "bias": self.layer2.bias
            },
            "config": {
                "l1_input_dim": self.l1_input_dim,
                "l1_res_dim": self.l1_res_dim,
                "l2_res_dim": self.l2_res_dim
            }
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Hierarchy saved to {path}")

    def load_hierarchy(self, path: str = "urcm_hierarchy.pkl"):
        """Loads weights."""
        if not os.path.exists(path):
            print(f"ℹ️ No hierarchy file found at {path}. Using random weights.")
            return
            
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                
            # Load Layer 1
            l1_data = data["layer1"]
            self.layer1.W_in = l1_data["W_in"]
            self.layer1.W_res = l1_data["W_res"]
            self.layer1.W_out = l1_data["W_out"]
            self.layer1.bias = l1_data["bias"]
            # Recalc inverse
            self.layer1.W_res_inv = np.linalg.pinv(self.layer1.W_res)
            
            # Load Layer 2
            l2_data = data["layer2"]
            self.layer2.W_in = l2_data["W_in"]
            self.layer2.W_res = l2_data["W_res"]
            self.layer2.W_out = l2_data["W_out"]
            self.layer2.bias = l2_data["bias"]
            # Recalc inverse
            self.layer2.W_res_inv = np.linalg.pinv(self.layer2.W_res)
            
            self.l2_is_trained = True
            print(f"✅ Hierarchy loaded from {path}")
            
        except Exception as e:
            print(f"❌ Error loading hierarchy: {e}")

from typing import Dict, List, Tuple

import numpy as np

from urcm.core.system import URCMSystem


class ConceptDecoder:
    """
    Decodes latent resonance states back into human-readable concepts
    using a Nearest Neighbor search over a pre-indexed vocabulary.
    This simulates 'decoding' without a heavy transformer decoder.
    """

    def __init__(self, system: URCMSystem):
        self.system = system
        self.vocab_vectors = []
        self.vocab_texts = []

    def build_index(self, concepts: List[str]):
        """
        Encodes a list of concepts and stores them for lookup.
        """
        print(f"Building Concept Index with {len(concepts)} items...")
        self.vocab_vectors = []
        self.vocab_texts = []

        for text in concepts:
            # Get the stable resonance state for this concept
            # We use the encoder directly
            path = self.system.pipeline.process_text(text)
            state = self.system.encoder.get_resonance_state(path)
            # Stabilize it slightly to match what the brain produces
            vec = state.resonance_vector
            # vec = self.system.gating.apply_gating(vec, dt=0.5)

            self.vocab_vectors.append(vec)
            self.vocab_texts.append(text)

        self.vocab_vectors = np.array(self.vocab_vectors)
        print("✅ Index built.")

    def decode(self, state_vector: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Finds the nearest concepts to the given state vector.
        Returns: List of (concept_text, similarity_score)
        """
        if len(self.vocab_vectors) == 0:
            return [("<<Empty Vocabulary>>", 0.0)]

        # Cosine Similarity
        # A . B / (|A| |B|)
        # Assume state_vector is normalized? Usually tanh outputs are not unit norm.

        # Normalize query
        q_norm = np.linalg.norm(state_vector) + 1e-9
        q = state_vector / q_norm

        # Normalize vocabulary (can be pre-computed but fast enough for demo)
        v_norms = np.linalg.norm(self.vocab_vectors, axis=1, keepdims=True) + 1e-9
        v = self.vocab_vectors / v_norms

        # Dot product
        sims = np.dot(v, q)

        # Top K
        indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in indices:
            results.append((self.vocab_texts[idx], float(sims[idx])))

        return results

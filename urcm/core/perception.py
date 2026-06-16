import numpy as np
from typing import Dict, List, Optional
from urcm.core.reasoning import ReasoningEngine

class PerceptionModule:
    """
    Handles Multi-Modal Perception for URCM.
    Since URCM is primarily a Semantic Vector System, this module
    translates raw sensory data (simulated) into Concept Vectors.
    """
    
    def __init__(self, engine: ReasoningEngine):
        self.engine = engine
        self.l2_dim = engine.l2_dim
        
        # Simulated Feature Extractors (would be CNN/RNN in real life)
        # We map "Hash of Image" -> "Concept Vector"
        self.visual_cortex: Dict[str, np.ndarray] = {}
        self.auditory_cortex: Dict[str, np.ndarray] = {}
        
    def process_image(self, image_id: str, tags: List[str] = None) -> np.ndarray:
        """
        Simulates seeing an image.
        If 'tags' are provided, we 'learn' to associate this image pattern with the tags.
        Returns the perception vector.
        """
        # 1. Generate a "Raw Sensory Vector" (simulated by hashing image_id)
        seed = sum(ord(c) for c in image_id) * 12345
        np.random.seed(seed % (2**32 - 1))
        sensory_vec = np.random.randn(self.l2_dim)
        sensory_vec = sensory_vec / np.linalg.norm(sensory_vec)
        
        # 2. If we have tags (Supervised Learning / Few-Shot),
        # we fuse the sensory vector with the concept vectors of the tags.
        if tags:
            concept_vecs = []
            for tag in tags:
                if tag in self.engine.concept_map:
                    concept_vecs.append(self.engine.concept_map[tag])
            
            if concept_vecs:
                # Mean of concepts
                semantic_target = np.mean(concept_vecs, axis=0)
                semantic_target = semantic_target / np.linalg.norm(semantic_target)
                
                # Hebbian Link: Rotate sensory vector towards semantic target
                # (Simulating "Learning what a cat looks like")
                # We store this "Learned Perception"
                # For simplicity in this simulation, we just store the mapping
                self.visual_cortex[image_id] = semantic_target
                return semantic_target
        
        # 3. If no tags, try to recognize (Recall)
        if image_id in self.visual_cortex:
            return self.visual_cortex[image_id]
            
        # 4. Unknown Image (Zero-Shot attempt? No, just raw sensation)
        return sensory_vec

    def process_audio(self, audio_id: str, transcript: str = None) -> np.ndarray:
        """
        Simulates hearing audio.
        """
        if transcript:
            # If we have transcript (like Whisper output), it's just text processing
            # We assume the audio system converts to text concepts
            words = transcript.lower().split()
            vecs = [self.engine.concept_map[w] for w in words if w in self.engine.concept_map]
            if vecs:
                avg_vec = np.mean(vecs, axis=0)
                avg_vec = avg_vec / np.linalg.norm(avg_vec)
                self.auditory_cortex[audio_id] = avg_vec
                return avg_vec
        
        return np.zeros(self.l2_dim)

    def describe_image(self, image_id: str) -> str:
        """
        Returns a text description of the image vector by finding nearest concepts.
        """
        vec = self.process_image(image_id)
        
        # Find nearest concept in engine
        best_word = "unknown"
        best_sim = -1.0
        
        for word, concept_vec in self.engine.concept_map.items():
            sim = np.dot(vec, concept_vec)
            if sim > best_sim:
                best_sim = sim
                best_word = word
                
        return best_word

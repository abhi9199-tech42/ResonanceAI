import os
import pickle
import re
from typing import Dict, List

import numpy as np

from urcm.core.hierarchical_encoder import HierarchicalEncoder
from urcm.core.identity import IDENTITY_CONCEPTS
from urcm.core.memory import GeometricMemory
from urcm.core.phoneme_mapper import TextToPhonemeConverter


class KnowledgeIngestion:
    """
    Ingests raw text into the URCM Resonance Memory.
    Enables 'Zero-Shot' World Knowledge and 'Open-Ended' Language capabilities
    by depositing sentence trajectories into the Concept Layer (L2).
    """

    def __init__(self, brain_path: str = "urcm_identity.pkl", l2_dim: int = 512):
        self.brain_path = brain_path
        self.l2_dim = l2_dim
        self.converter = TextToPhonemeConverter()
        self.memory = GeometricMemory(resonance_dim=l2_dim) # Increased capacity

        # Load or Create Brain
        if os.path.exists(brain_path):
            print(f"Loading brain from {brain_path}...")
            from urcm.core.safe_io import safe_load_pickle
            self.brain_data = safe_load_pickle(brain_path)
            # Check dim
            if self.brain_data["l2_W_res"].shape[0] != l2_dim:
                print(f"⚠️ Resizing Brain from {self.brain_data['l2_W_res'].shape[0]} to {l2_dim}")
                # Re-init
                self._init_fresh_brain()
        else:
            print("Creating FRESH High-Capacity Brain...")
            self._init_fresh_brain()

        # Reconstruct Hierarchy State
        self.hierarchy = HierarchicalEncoder(l2_res_dim=l2_dim)
        self.hierarchy.layer2.W_res = self.brain_data["l2_W_res"]
        self.concept_map = self.brain_data["concept_map"]

    def _init_fresh_brain(self):
        """Initializes a new brain with Identity concepts."""
        hierarchy = HierarchicalEncoder(l2_res_dim=self.l2_dim)

        # Init Concept Map
        concept_map = {}
        self.concept_map = concept_map # Set temporarily for helper use

        # Initialize Identity Concepts
        print("Initializing Identity Concepts...")
        # Populate map with Identity Concepts
        for concept_name in IDENTITY_CONCEPTS.keys():
            self._get_or_create_concept_vector(concept_name)

        self.brain_data = {
            "l1_W_res": hierarchy.layer1.W_res,
            "l1_W_in": hierarchy.layer1.W_in,
            "l1_W_out": hierarchy.layer1.W_out,
            "l2_W_res": hierarchy.layer2.W_res, # 512x512
            "l2_W_in": hierarchy.layer2.W_in,
            "l2_W_out": hierarchy.layer2.W_out,
            "concept_map": concept_map
        }

    def _get_or_create_concept_vector(self, word: str) -> np.ndarray:
        """
        Returns the L2 vector for a word.
        If the word is new, generates a deterministic vector and adds it to the map.
        """
        word = word.lower()

        # 1. Check existing
        if word in self.concept_map:
            return self.concept_map[word]

        # 2. Create new (Deterministic Hash)
        # Using 512 dimensions for high capacity separation
        # Phase 10 Fix: Improved Hashing to prevent collisions (e.g. midway vs across)
        import hashlib
        hash_object = hashlib.md5(word.encode())
        seed = int(hash_object.hexdigest(), 16) % (2**32)

        rng = np.random.RandomState(seed)
        vec = rng.normal(0, 1, (self.l2_dim,))
        vec = vec / np.linalg.norm(vec)

        self.concept_map[word] = vec
        return vec

    def ingest_text(self, text: str):
        """
        Parses text and deposits semantic trajectories.
        """
        # 1. Clean and Split into Sentences
        text = re.sub(r'\s+', ' ', text) # Normalize whitespace
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip().lower() for s in sentences if s.strip()]

        # Phase 9 Fix: Semantic Gating (Black Hole Suppression)
        # We actively filter out words that cause attractor collapse.
        STOP_WORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "has", "have", "had", "having", "do", "does", "did", "doing",
            "can", "could", "would", "should", "will", "shall", "may", "might", "must",
            "to", "of", "and", "in", "that", "this", "it", "for", "on", "with",
            "as", "by", "at", "from", "or", "but", "not", "if", "then", "else",
            "help", "causing", "copyright", "privacy", "disclaimer", "contact",
            "hours", "minute", "minutes", "second", "seconds", "day", "days",
            "wiki", "edit", "source", "history", "links", "external", "references",
            "contents", "search", "navigation", "main", "page", "article", "talk",
            "create", "account", "log", "in", "out", "view", "read", "jump", "content",
            "file", "image", "category", "template", "user", "portal", "special",
            "identifier", "identifiers", "link", "about", "all", "any", "some",
            "retrieved", "archived", "original", "date", "isbn", "doi", "issn", "pmid",
            "text", "under", "license", "additional", "terms", "apply", "site",
            "policy", "mobile", "developers", "statistics", "cookie", "statement",
            "wikimedia", "foundation", "powered", "mediawiki",
            "level", "levels", "type", "types", "use", "used", "using", # New Grey Holes
        }

        print(f"Ingesting {len(sentences)} sentences into {self.l2_dim}-dim Space...")

        count = 0
        for sentence in sentences:
            # Tokenize
            words = re.findall(r'\w+', sentence)

            # Apply Filter
            filtered_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]

            if len(filtered_words) < 2:
                continue # Skip empty/short sentences

            # DEBUG: Print words being deposited
            # if "midway" in filtered_words:
            #    print(f"DEBUG: Depositing sequence with 'midway': {filtered_words}")

            self.deposit_sequence(filtered_words)
            count += 1

        print(f"✅ Successfully ingested {count} knowledge trajectories.")

    def deposit_sequence(self, words: List[str]):
        trajectory = [self._get_or_create_concept_vector(w) for w in words]
        self.hierarchy.layer2.W_res = self.memory.deposit_sequence(
            self.hierarchy.layer2.W_res,
            trajectory
        )

    def ingest_file(self, file_path: str):
        """Smart ingestion based on file type."""
        print(f"Reading {file_path}...")

        if file_path.endswith(".json"):
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Extract text from known structures (list of dicts)
            text = ""
            if isinstance(data, list):
                for item in data:
                    context_prefix = ""
                    if "scenario_id" in item:
                        # Extract key terms from ID (e.g. WW2_PACIFIC_1942 -> ww2 pacific)
                        context_prefix = item["scenario_id"].replace("_", " ") + " "

                    if "description" in item:
                        text += context_prefix + item["description"] + ". "

                    if "intents" in item:
                        for intent in item["intents"]:
                            if "description" in intent:
                                # Bind Intent to Context
                                # Include ID for keywords (e.g. AMBUSH_CARRIERS -> ambush carriers)
                                intent_keywords = intent["id"].replace("_", " ") if "id" in intent else ""
                                text += context_prefix + intent_keywords + " " + intent["description"] + ". "
            self.ingest_text(text)

        elif file_path.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Simple Markdown cleanup
            text = re.sub(r'[#*`]', '', content)
            self.ingest_text(text)

        else:
            # Default text
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.ingest_text(text)

    def save(self):
        """Saves the updated brain."""
        self.brain_data["l2_W_res"] = self.hierarchy.layer2.W_res
        self.brain_data["concept_map"] = self.concept_map

        with open(self.brain_path, "wb") as f:
            pickle.dump(self.brain_data, f)
        print(f"Brain saved to {self.brain_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest text into URCM memory.")
    parser.add_argument("file", help="Text file to ingest")
    args = parser.parse_args()

    # Use 512 dimensions for better capacity
    ingestor = KnowledgeIngestion(l2_dim=512)

    ingestor.ingest_file(args.file)
    ingestor.save()

if __name__ == "__main__":
    main()

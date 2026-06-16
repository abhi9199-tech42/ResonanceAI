from typing import Dict, List, Optional

class SanskritBridge:
    """
    Bridge between English concepts and Sanskrit philosophical terminology.
    Provides mapping and IAST transliteration for URCM resonance states.
    """
    
    # Core philosophical dictionary (English -> Sanskrit IAST)
    CONCEPT_MAP: Dict[str, str] = {
        # Ontology / Existence
        "consciousness": "cit",
        "existence": "sat",
        "being": "sat",
        "truth": "satya",
        "illusion": "māyā",
        "void": "śūnya",
        "absolute": "brahman",
        "self": "ātman",
        "soul": "ātman",
        "world": "jagat",
        "universe": "brahmāṇḍa",
        
        # Dynamics / Energy
        "energy": "śakti",
        "power": "śakti",
        "force": "bala",
        "movement": "gati",
        "action": "karma",
        "cause": "kāraṇa",
        "effect": "kārya",
        "change": "pariṇāma",
        "flow": "pravāha",
        "vibration": "spanda",
        "resonance": "dhvani",
        
        # Cognition / Mind
        "mind": "manas",
        "intellect": "buddhi",
        "ego": "ahaṃkāra",
        "thought": "vṛtti",
        "knowledge": "jñāna",
        "wisdom": "prajñā",
        "ignorance": "avidyā",
        "memory": "smṛti",
        "perception": "pratyakṣa",
        "inference": "anumāna",
        
        # Ethics / Affect
        "peace": "śānti",
        "joy": "ānanda",
        "bliss": "ānanda",
        "suffering": "duḥkha",
        "desire": "kāma",
        "liberation": "mokṣa",
        "freedom": "svātantrya",
        "dharma": "dharma",
        "law": "dharma",
        "compassion": "karuṇā",
        "love": "prema",
        
        # Strategic / Conflict (for Midway scenarios)
        "war": "yuddha",
        "battle": "saṅgrāma",
        "victory": "vijaya",
        "defeat": "parājaya",
        "enemy": "śatru",
        "ally": "mitra",
        "king": "rājan",
        "leader": "netṛ",
        "strategy": "upāya",
        "plan": "yojanā",
        "weapon": "astra",
        "attack": "ākramaṇa",
        "defense": "rakṣā"
    }

    def __init__(self):
        self.reverse_map = {v: k for k, v in self.CONCEPT_MAP.items()}

    def get_sanskrit(self, english_term: str) -> Optional[str]:
        """
        Get the Sanskrit IAST equivalent for an English concept.
        Returns None if no mapping exists.
        """
        term = english_term.lower().strip()
        return self.CONCEPT_MAP.get(term)

    def get_english(self, sanskrit_term: str) -> Optional[str]:
        """
        Get the English equivalent for a Sanskrit term.
        """
        term = sanskrit_term.lower().strip()
        return self.reverse_map.get(term)

    def translate_trajectory(self, trajectory: List[str]) -> List[str]:
        """
        Augment a trajectory of English concepts with their Sanskrit equivalents.
        Format: "concept [sanskrit]"
        """
        translated = []
        for concept in trajectory:
            sanskrit = self.get_sanskrit(concept)
            if sanskrit:
                translated.append(f"{concept} [{sanskrit}]")
            else:
                translated.append(concept)
        return translated

    def get_resonance_signature(self, concept: str) -> str:
        """
        Returns a string representation of the concept's philosophical resonance.
        Useful for logging or UI display.
        """
        sanskrit = self.get_sanskrit(concept)
        if sanskrit:
            return f"{concept.upper()} :: {sanskrit.upper()}"
        return concept.upper()

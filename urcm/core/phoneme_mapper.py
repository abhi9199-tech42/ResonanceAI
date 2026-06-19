"""
Phoneme-to-frequency mapping system for the URCM reasoning system.

This module implements the core phoneme-frequency mapping functionality that converts
Sanskrit-derived phonemes into continuous frequency vectors in K-dimensional space.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .data_models import FrequencyPath, PhonemeSequence


class PhonemeFrequencyMapper:
    """
    Maps Sanskrit-derived phonemes to continuous frequency vectors in K-dimensional space.

    This class implements the core transformation from discrete phonemes to continuous
    frequency representations, enforcing smoothness constraints and maintaining
    consistency across mappings.
    """

    # Sanskrit-derived phoneme set for complete articulatory coverage
    SANSKRIT_PHONEMES = {
        # Vowels (स्वर)
        'a', 'ā', 'i', 'ī', 'u', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ', 'e', 'ai', 'o', 'au',

        # Consonants - Stops (स्पर्श)
        # Velar (कण्ठ्य)
        'k', 'kh', 'g', 'gh', 'ṅ',
        # Palatal (तालव्य)
        'c', 'ch', 'j', 'jh', 'ñ',
        # Retroflex (मूर्धन्य)
        'ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ',
        # Dental (दन्त्य)
        't', 'th', 'd', 'dh', 'n',
        # Labial (ओष्ठ्य)
        'p', 'ph', 'b', 'bh', 'm',

        # Semivowels (अन्तःस्थ)
        'y', 'r', 'l', 'v',

        # Sibilants (ऊष्म)
        'ś', 'ṣ', 's', 'h',

        # Additional phonemes for broader coverage
        'x', 'z', 'f', 'w',  # For non-Sanskrit languages
        'SIL'  # Word boundary / silence token
    }

    def __init__(self, frequency_dim: int = 24, smoothness_weight: float = 0.1):
        """
        Initialize the phoneme-frequency mapper.

        Args:
            frequency_dim: Dimensionality of frequency vectors (K ∈ [16, 32])
            smoothness_weight: Weight for smoothness constraint enforcement
        """
        if not (16 <= frequency_dim <= 32):
            raise ValueError("Frequency dimension K must be in range [16, 32]")

        self.phoneme_space = self.SANSKRIT_PHONEMES.copy()
        self.K = frequency_dim
        self.smoothness_weight = smoothness_weight

        # Initialize phoneme-to-frequency mapping
        self._initialize_phoneme_vectors()

        # Articulatory feature matrix for smoothness constraints
        self._initialize_articulatory_features()

    def _initialize_phoneme_vectors(self):
        """Initialize deterministic phoneme-to-frequency vector mappings."""
        # Create deterministic mapping based on phoneme characteristics
        self.phoneme_vectors = {}

        # Use a deterministic seed for consistent mappings
        rng = np.random.RandomState(42)

        # Group phonemes by articulatory features for better organization
        phoneme_groups = self._group_phonemes_by_features()

        # Generate vectors with structured initialization
        for group_name, phonemes in phoneme_groups.items():
            # Each group gets a base vector in a different region of frequency space
            group_base = rng.uniform(-1, 1, self.K)
            group_base = group_base / np.linalg.norm(group_base)  # Normalize

            for i, phoneme in enumerate(phonemes):
                # Add small perturbations within the group
                perturbation = rng.uniform(-0.3, 0.3, self.K)
                vector = group_base + perturbation
                # Normalize to unit sphere for consistency
                vector = vector / np.linalg.norm(vector)
                self.phoneme_vectors[phoneme] = vector

    def _group_phonemes_by_features(self) -> Dict[str, List[str]]:
        """Group phonemes by articulatory features for structured mapping."""
        groups = {
            'vowels': ['a', 'ā', 'i', 'ī', 'u', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ', 'e', 'ai', 'o', 'au'],
            'velars': ['k', 'kh', 'g', 'gh', 'ṅ'],
            'palatals': ['c', 'ch', 'j', 'jh', 'ñ'],
            'retroflexes': ['ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ'],
            'dentals': ['t', 'th', 'd', 'dh', 'n'],
            'labials': ['p', 'ph', 'b', 'bh', 'm'],
            'semivowels': ['y', 'r', 'l', 'v'],
            'sibilants': ['ś', 'ṣ', 's', 'h'],
            'additional': ['x', 'z', 'f', 'w']
        }

        # Filter to only include phonemes that exist in our set
        filtered_groups = {}
        for group_name, phonemes in groups.items():
            filtered_phonemes = [p for p in phonemes if p in self.phoneme_space]
            if filtered_phonemes:
                filtered_groups[group_name] = filtered_phonemes

        # Ensure SIL gets a vector if it's in the phoneme space
        if 'SIL' in self.phoneme_space and 'SIL' not in [p for ps in filtered_groups.values() for p in ps]:
            filtered_groups['boundary'] = ['SIL']

        return filtered_groups

    def _initialize_articulatory_features(self):
        """Initialize articulatory feature matrix for smoothness calculations."""
        # Simplified articulatory features for smoothness constraints
        # Features: [vowel, voiced, aspirated, nasal, retroflex, palatal, velar, dental, labial]
        self.articulatory_features = {}

        # Define features for each phoneme (simplified)
        feature_definitions = {
            # Vowels
            'a': [1, 1, 0, 0, 0, 0, 0, 0, 0], 'ā': [1, 1, 0, 0, 0, 0, 0, 0, 0],
            'i': [1, 1, 0, 0, 0, 1, 0, 0, 0], 'ī': [1, 1, 0, 0, 0, 1, 0, 0, 0],
            'u': [1, 1, 0, 0, 0, 0, 0, 0, 1], 'ū': [1, 1, 0, 0, 0, 0, 0, 0, 1],
            'e': [1, 1, 0, 0, 0, 1, 0, 0, 0], 'o': [1, 1, 0, 0, 0, 0, 0, 0, 1],
            'ai': [1, 1, 0, 0, 0, 1, 0, 0, 0], 'au': [1, 1, 0, 0, 0, 0, 0, 0, 1],
            'ṛ': [1, 1, 0, 0, 1, 0, 0, 0, 0], 'ṝ': [1, 1, 0, 0, 1, 0, 0, 0, 0],
            'ḷ': [1, 1, 0, 0, 1, 0, 0, 0, 0], 'ḹ': [1, 1, 0, 0, 1, 0, 0, 0, 0],

            # Velars
            'k': [0, 0, 0, 0, 0, 0, 1, 0, 0], 'kh': [0, 0, 1, 0, 0, 0, 1, 0, 0],
            'g': [0, 1, 0, 0, 0, 0, 1, 0, 0], 'gh': [0, 1, 1, 0, 0, 0, 1, 0, 0],
            'ṅ': [0, 1, 0, 1, 0, 0, 1, 0, 0],

            # Palatals
            'c': [0, 0, 0, 0, 0, 1, 0, 0, 0], 'ch': [0, 0, 1, 0, 0, 1, 0, 0, 0],
            'j': [0, 1, 0, 0, 0, 1, 0, 0, 0], 'jh': [0, 1, 1, 0, 0, 1, 0, 0, 0],
            'ñ': [0, 1, 0, 1, 0, 1, 0, 0, 0],

            # Retroflexes
            'ṭ': [0, 0, 0, 0, 1, 0, 0, 0, 0], 'ṭh': [0, 0, 1, 0, 1, 0, 0, 0, 0],
            'ḍ': [0, 1, 0, 0, 1, 0, 0, 0, 0], 'ḍh': [0, 1, 1, 0, 1, 0, 0, 0, 0],
            'ṇ': [0, 1, 0, 1, 1, 0, 0, 0, 0],

            # Dentals
            't': [0, 0, 0, 0, 0, 0, 0, 1, 0], 'th': [0, 0, 1, 0, 0, 0, 0, 1, 0],
            'd': [0, 1, 0, 0, 0, 0, 0, 1, 0], 'dh': [0, 1, 1, 0, 0, 0, 0, 1, 0],
            'n': [0, 1, 0, 1, 0, 0, 0, 1, 0],

            # Labials
            'p': [0, 0, 0, 0, 0, 0, 0, 0, 1], 'ph': [0, 0, 1, 0, 0, 0, 0, 0, 1],
            'b': [0, 1, 0, 0, 0, 0, 0, 0, 1], 'bh': [0, 1, 1, 0, 0, 0, 0, 0, 1],
            'm': [0, 1, 0, 1, 0, 0, 0, 0, 1],

            # Semivowels
            'y': [0, 1, 0, 0, 0, 1, 0, 0, 0], 'r': [0, 1, 0, 0, 1, 0, 0, 0, 0],
            'l': [0, 1, 0, 0, 0, 0, 0, 1, 0], 'v': [0, 1, 0, 0, 0, 0, 0, 0, 1],

            # Sibilants
            'ś': [0, 0, 0, 0, 0, 1, 0, 0, 0], 'ṣ': [0, 0, 0, 0, 1, 0, 0, 0, 0],
            's': [0, 0, 0, 0, 0, 0, 0, 1, 0], 'h': [0, 1, 1, 0, 0, 0, 1, 0, 0],

            # Additional
            'x': [0, 0, 0, 0, 0, 0, 1, 0, 0], 'z': [0, 1, 0, 0, 0, 0, 0, 1, 0],
            'f': [0, 0, 0, 0, 0, 0, 0, 0, 1], 'w': [0, 1, 0, 0, 0, 0, 0, 0, 1]
        }

        for phoneme, features in feature_definitions.items():
            if phoneme in self.phoneme_space:
                self.articulatory_features[phoneme] = np.array(features, dtype=float)

    def map_phoneme(self, phoneme: str) -> np.ndarray:
        """
        Map single phoneme to frequency vector.

        Args:
            phoneme: Sanskrit phoneme to map

        Returns:
            K-dimensional frequency vector

        Raises:
            ValueError: If phoneme is not in the Sanskrit phoneme set
        """
        if phoneme not in self.phoneme_space:
            raise ValueError(f"Phoneme '{phoneme}' not in Sanskrit phoneme set")

        return self.phoneme_vectors[phoneme].copy()

    def map_sequence(self, phonemes: List[str]) -> np.ndarray:
        """
        Map phoneme sequence to frequency path.

        Args:
            phonemes: List of Sanskrit phonemes

        Returns:
            Frequency path matrix of shape (len(phonemes), K)

        Raises:
            ValueError: If any phoneme is not in the Sanskrit phoneme set
        """
        if not phonemes:
            raise ValueError("Phoneme sequence cannot be empty")

        # Map each phoneme to its frequency vector
        vectors = []
        for phoneme in phonemes:
            vector = self.map_phoneme(phoneme)
            vectors.append(vector)

        frequency_path = np.array(vectors)

        # Apply smoothness constraints if sequence has multiple phonemes
        if len(phonemes) > 1:
            frequency_path = self.enforce_smoothness(frequency_path)

        return frequency_path

    def enforce_smoothness(self, path: np.ndarray) -> np.ndarray:
        """
        Apply smoothness constraints to frequency path.

        This method minimizes ||f(pi) - f(pj)||² for adjacent phonemes
        while preserving the overall semantic structure.

        Args:
            path: Original frequency path of shape (sequence_length, K)

        Returns:
            Smoothed frequency path with same shape
        """
        if path.shape[0] <= 1:
            return path.copy()

        smoothed_path = path.copy()

        # Apply iterative smoothing
        for iteration in range(3):  # Limited iterations to prevent over-smoothing
            for i in range(1, len(smoothed_path) - 1):
                # Calculate smoothing adjustment based on neighbors
                prev_vec = smoothed_path[i - 1]
                curr_vec = smoothed_path[i]
                next_vec = smoothed_path[i + 1]

                # Smoothing: move current vector towards average of neighbors
                neighbor_avg = (prev_vec + next_vec) / 2
                adjustment = (neighbor_avg - curr_vec) * self.smoothness_weight

                # Apply adjustment while preserving vector magnitude
                new_vec = curr_vec + adjustment
                # Normalize to maintain consistent magnitude
                new_vec = new_vec / np.linalg.norm(new_vec) * np.linalg.norm(curr_vec)

                smoothed_path[i] = new_vec

        return smoothed_path

    def calculate_smoothness_score(self, path: np.ndarray) -> float:
        """
        Calculate smoothness score for a frequency path.

        Args:
            path: Frequency path of shape (sequence_length, K)

        Returns:
            Smoothness score (lower values indicate smoother paths)
        """
        if path.shape[0] <= 1:
            return 0.0

        # Calculate pairwise distances between adjacent vectors
        diffs = np.diff(path, axis=0)
        distances = np.linalg.norm(diffs, axis=1)

        # Smoothness score is the average distance between adjacent vectors
        smoothness_score = np.mean(distances)

        return float(smoothness_score)

    def create_frequency_path(self, phonemes: List[str]) -> FrequencyPath:
        """
        Create a complete FrequencyPath object from phoneme sequence.

        Args:
            phonemes: List of Sanskrit phonemes

        Returns:
            FrequencyPath object with vectors, smoothness score, and mapping
        """
        if not phonemes:
            raise ValueError("Phoneme sequence cannot be empty")

        # Map phonemes to frequency vectors
        vectors = self.map_sequence(phonemes)

        # Calculate smoothness score
        smoothness_score = self.calculate_smoothness_score(vectors)

        # Create phoneme mapping
        phoneme_mapping = [(phoneme, i) for i, phoneme in enumerate(phonemes)]

        return FrequencyPath(
            vectors=vectors,
            smoothness_score=smoothness_score,
            phoneme_mapping=phoneme_mapping
        )


class TextToPhonemeConverter:
    """
    Converts text input to phoneme sequences for frequency mapping.

    Supports both English (approximate) and Sanskrit (IAST) inputs.
    """

    def __init__(self):
        """Initialize the text-to-phoneme converter."""
        # English -> Approximate Sanskrit Phoneme Map
        self.english_map = {
            'a': 'a', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'e', 'f': 'f',
            'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l',
            'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 'q': 'k', 'r': 'r',
            's': 's', 't': 't', 'u': 'u', 'v': 'v', 'w': 'v', 'x': 'x',
            'y': 'y', 'z': 'z',
        }
        # Common English digraphs
        self.english_digraphs = {
            'sh': 'ś', 'ch': 'c', 'zh': 'z', 'ee': 'ī', 'oo': 'ū',
            'th': 't', 'kh': 'kh', 'gh': 'gh', 'ph': 'f', 'wh': 'v',
        }

        # Sanskrit IAST Direct Map (Identity for known phonemes)
        # We also handle some common transliteration variants
        self.sanskrit_map = {
            'ā': 'ā', 'ī': 'ī', 'ū': 'ū',
            'ṛ': 'ṛ', 'ṝ': 'ṝ', 'ḷ': 'ḷ', 'ḹ': 'ḹ',
            'ai': 'ai', 'au': 'au',
            'ṅ': 'ṅ', 'ñ': 'ñ', 'ṇ': 'ṇ',
            'ṭ': 'ṭ', 'ṭh': 'ṭh', 'ḍ': 'ḍ', 'ḍh': 'ḍh',
            'ś': 'ś', 'ṣ': 'ṣ', 'ḥ': 'h', 'ṃ': 'm'
        }

    def convert_text_to_phonemes(self, text: str, language_hint: Optional[str] = None) -> PhonemeSequence:
        """
        Convert text to phoneme sequence.

        Detects if input contains IAST characters to switch to Sanskrit mode.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        text = text.strip().lower()
        phonemes = []

        # 1. Tokenize (Simple greedy match for multi-char phonemes like 'ai', 'au', 'kh', 'gh'...)
        # We need to process from left to right, matching longest phonemes first.
        # This is crucial for aspirated consonants (kh, gh) vs k+h.

        # Pre-scan for IAST characters to detect Sanskrit mode
        has_iast = any(c in 'āīūṛṝḷḹṅñṇṭḍḍhśṣḥṃ' for c in text)
        # Use language_hint or auto-detect
        is_sanskrit = has_iast or language_hint == 'sa'

        i = 0
        while i < len(text):
            # Try 2-char match first
            if i + 1 < len(text):
                chunk = text[i:i+2]
                if not is_sanskrit and chunk in self.english_digraphs:
                    phonemes.append(self.english_digraphs[chunk])
                    i += 2
                    continue
                if chunk in ['ai', 'au', 'kh', 'gh', 'ch', 'jh', 'th', 'dh', 'ph', 'bh']:
                    phonemes.append(chunk)
                    i += 2
                    continue
                if is_sanskrit and chunk in self.english_digraphs:
                    phonemes.append(self.english_digraphs[chunk])
                    i += 2
                    continue

            # Single char match
            char = text[i]

            # Check Sanskrit specific
            if char in self.sanskrit_map:
                phonemes.append(self.sanskrit_map[char])
            # Check English/Standard
            elif char in self.english_map:
                phonemes.append(self.english_map[char])
            # Handle space/punctuation — add word boundary token
            elif char in [' ', ',', '.', '-']:
                 if phonemes and phonemes[-1] != 'SIL':
                     phonemes.append('SIL')

            i += 1

        if not phonemes:
            phonemes = ['a']

        return PhonemeSequence(
            phonemes=phonemes,
            source_text=text,
            language_hint="sanskrit" if any(c in self.sanskrit_map for c in text) else "english"
        )


class PhonemeFrequencyPipeline:
    """
    Complete pipeline for text-to-frequency conversion.

    This class combines text-to-phoneme conversion with phoneme-to-frequency
    mapping to provide a complete processing pipeline.
    """

    def __init__(self, frequency_dim: int = 24, smoothness_weight: float = 0.1):
        """
        Initialize the complete pipeline.

        Args:
            frequency_dim: Dimensionality of frequency vectors (K ∈ [16, 32])
            smoothness_weight: Weight for smoothness constraint enforcement
        """
        self.text_converter = TextToPhonemeConverter()
        self.frequency_mapper = PhonemeFrequencyMapper(frequency_dim, smoothness_weight)

    def process_text(self, text: str, language_hint: Optional[str] = None) -> FrequencyPath:
        """
        Process text through complete pipeline: text → phonemes → frequency path.

        Args:
            text: Input text to process
            language_hint: Optional language hint for better conversion

        Returns:
            FrequencyPath object representing the text in frequency space
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        # Convert text to phonemes
        phoneme_sequence = self.text_converter.convert_text_to_phonemes(text, language_hint)

        # Convert phonemes to frequency path
        frequency_path = self.frequency_mapper.create_frequency_path(phoneme_sequence.phonemes)

        return frequency_path

    def validate_pipeline(self, text: str) -> bool:
        """
        Validate that the complete pipeline produces valid output.

        Args:
            text: Test text to validate

        Returns:
            True if pipeline produces valid FrequencyPath, False otherwise
        """
        try:
            frequency_path = self.process_text(text)
            from .validation import DataValidation
            return DataValidation.validate_frequency_path(frequency_path)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("validate_transcription exception: %s", e)
            return False

"""
Performance optimization and efficiency monitoring for URCM.

This module provides:
- Memory-efficient phoneme set management
- Compression efficiency monitoring
- Performance benchmarking against token-based systems
- Computational efficiency tracking
"""

import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PerformanceMetrics:
    """Tracks performance metrics for URCM operations."""

    # Memory metrics
    phoneme_set_size: int = 0
    frequency_vector_dimension: int = 0
    total_memory_bytes: int = 0

    # Compression metrics
    input_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 0.0

    # Computational metrics
    processing_time_ms: float = 0.0
    operations_count: int = 0

    # Comparison with token-based systems
    token_equivalent_size: int = 0
    token_memory_bytes: int = 0
    memory_efficiency_ratio: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.input_size_bytes > 0 and self.compressed_size_bytes > 0:
            self.compression_ratio = self.input_size_bytes / self.compressed_size_bytes

        if self.token_memory_bytes > 0 and self.total_memory_bytes > 0:
            self.memory_efficiency_ratio = self.token_memory_bytes / self.total_memory_bytes


@dataclass
class OptimizedPhonemeSet:
    """
    Memory-efficient phoneme set management.

    Uses compact representations and caching to minimize memory footprint
    while maintaining fast lookup performance.
    """

    # Core phoneme mapping (compact storage)
    _phoneme_to_id: Dict[str, int] = field(default_factory=dict)
    _id_to_phoneme: List[str] = field(default_factory=list)

    # Frequency vector cache (lazy loading)
    _frequency_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    _cache_hits: int = 0
    _cache_misses: int = 0

    # Configuration
    vector_dimension: int = 24  # K ∈ [16, 32]
    max_cache_size: int = 1000

    def __post_init__(self):
        """Initialize with Sanskrit-derived phoneme set."""
        if not self._phoneme_to_id:
            self._initialize_phoneme_set()

    def _initialize_phoneme_set(self):
        """Initialize compact phoneme set (finite, small)."""
        # Sanskrit-derived phonemes (compact representation)
        phonemes = [
            # Vowels (स्वर)
            'a', 'ā', 'i', 'ī', 'u', 'ū', 'ṛ', 'ṝ', 'ḷ', 'e', 'ai', 'o', 'au',

            # Stops (स्पर्श)
            'k', 'kh', 'g', 'gh', 'ṅ',  # Velar
            'c', 'ch', 'j', 'jh', 'ñ',  # Palatal
            'ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ',  # Retroflex
            't', 'th', 'd', 'dh', 'n',  # Dental
            'p', 'ph', 'b', 'bh', 'm',  # Labial

            # Approximants (अन्तस्थ)
            'y', 'r', 'l', 'v',

            # Fricatives (ऊष्म)
            'ś', 'ṣ', 's', 'h'
        ]

        for idx, phoneme in enumerate(phonemes):
            self._phoneme_to_id[phoneme] = idx
            self._id_to_phoneme.append(phoneme)

    def get_phoneme_id(self, phoneme: str) -> Optional[int]:
        """Get compact integer ID for phoneme."""
        return self._phoneme_to_id.get(phoneme)

    def get_phoneme(self, phoneme_id: int) -> Optional[str]:
        """Get phoneme from compact ID."""
        if 0 <= phoneme_id < len(self._id_to_phoneme):
            return self._id_to_phoneme[phoneme_id]
        return None

    def get_frequency_vector(self, phoneme: str, use_cache: bool = True) -> np.ndarray:
        """
        Get frequency vector for phoneme with caching.

        Args:
            phoneme: Phoneme string
            use_cache: Whether to use cache (default True)

        Returns:
            K-dimensional frequency vector
        """
        phoneme_id = self.get_phoneme_id(phoneme)
        if phoneme_id is None:
            raise ValueError(f"Unknown phoneme: {phoneme}")

        # Check cache
        if use_cache and phoneme_id in self._frequency_cache:
            self._cache_hits += 1
            return self._frequency_cache[phoneme_id]

        self._cache_misses += 1

        # Generate frequency vector (deterministic based on phoneme properties)
        vector = self._generate_frequency_vector(phoneme, phoneme_id)

        # Cache management
        if use_cache:
            if len(self._frequency_cache) >= self.max_cache_size:
                # FIFO eviction (dict insertion order)
                self._frequency_cache.pop(next(iter(self._frequency_cache)))
            self._frequency_cache[phoneme_id] = vector

        return vector

    def _generate_frequency_vector(self, phoneme: str, phoneme_id: int) -> np.ndarray:
        """
        Generate K-dimensional frequency vector for phoneme.

        Uses deterministic generation based on phoneme articulatory properties.
        """
        # Seed based on phoneme ID for reproducibility
        rng = np.random.RandomState(phoneme_id)

        # Generate base frequencies
        base_freq = rng.randn(self.vector_dimension)

        # Normalize to unit sphere (compact representation)
        base_freq = base_freq / (np.linalg.norm(base_freq) + 1e-9)

        # Scale to reasonable range [0, 1]
        vector = (base_freq + 1.0) / 2.0

        return vector.astype(np.float32)  # Use float32 for memory efficiency

    def get_memory_usage(self) -> int:
        """Calculate total memory usage in bytes."""
        # Phoneme mapping
        mapping_size = sys.getsizeof(self._phoneme_to_id) + sys.getsizeof(self._id_to_phoneme)

        # Cache size
        cache_size = sum(
            v.nbytes for v in self._frequency_cache.values()
        ) + sys.getsizeof(self._frequency_cache)

        return mapping_size + cache_size

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._frequency_cache),
            'max_cache_size': self.max_cache_size
        }

    @property
    def size(self) -> int:
        """Number of phonemes in the set."""
        return len(self._id_to_phoneme)


class CompressionMonitor:
    """
    Monitors compression efficiency of semantic latent space.

    Tracks compression ratios and ensures efficiency constraints are met.
    """

    def __init__(self):
        self.compression_history: List[Tuple[float, float]] = []  # (input_size, compressed_size)
        self.efficiency_threshold: float = 2.0  # Minimum compression ratio

    def record_compression(self, input_size: float, compressed_size: float):
        """Record a compression operation."""
        self.compression_history.append((input_size, compressed_size))

    def get_average_compression_ratio(self) -> float:
        """Calculate average compression ratio."""
        if not self.compression_history:
            return 0.0

        ratios = [inp / comp for inp, comp in self.compression_history if comp > 0]
        return np.mean(ratios) if ratios else 0.0

    def get_compression_efficiency(self) -> Dict[str, float]:
        """Get detailed compression efficiency metrics."""
        if not self.compression_history:
            return {
                'average_ratio': 0.0,
                'min_ratio': 0.0,
                'max_ratio': 0.0,
                'std_ratio': 0.0,
                'meets_threshold': False
            }

        ratios = [inp / comp for inp, comp in self.compression_history if comp > 0]

        return {
            'average_ratio': np.mean(ratios),
            'min_ratio': np.min(ratios),
            'max_ratio': np.max(ratios),
            'std_ratio': np.std(ratios),
            'meets_threshold': np.mean(ratios) >= self.efficiency_threshold
        }

    def validate_efficiency(self) -> bool:
        """Validate that compression meets efficiency threshold."""
        avg_ratio = self.get_average_compression_ratio()
        return avg_ratio >= self.efficiency_threshold


class PerformanceBenchmark:
    """
    Benchmarks URCM performance against token-based systems.

    Compares memory usage, processing speed, and compression efficiency.
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def benchmark_memory_efficiency(
        self,
        text: str,
        phoneme_set: OptimizedPhonemeSet,
        latent_dim: int = 128
    ) -> Dict[str, Any]:
        """
        Compare memory usage: URCM vs token-based systems.

        Args:
            text: Input text to process
            phoneme_set: Optimized phoneme set
            latent_dim: Latent space dimension

        Returns:
            Comparison metrics
        """
        # URCM memory calculation
        # 1. Phoneme set storage
        phoneme_memory = phoneme_set.get_memory_usage()

        # 2. Frequency vectors (K-dimensional, float32)
        text_length = len(text)
        freq_vector_memory = text_length * phoneme_set.vector_dimension * 4  # 4 bytes per float32

        # 3. Latent space (compressed)
        latent_memory = latent_dim * 4  # Single latent vector

        urcm_total = phoneme_memory + freq_vector_memory + latent_memory

        # Token-based system memory (e.g., GPT-style)
        # Vocabulary size: ~50k tokens, embedding dim: 768
        vocab_size = 50000
        embedding_dim = 768
        token_vocab_memory = vocab_size * embedding_dim * 4  # Embedding table
        token_sequence_memory = text_length * embedding_dim * 4  # Sequence embeddings

        token_total = token_vocab_memory + token_sequence_memory

        # Calculate efficiency
        memory_ratio = token_total / urcm_total if urcm_total > 0 else 0

        results = {
            'urcm_memory_bytes': urcm_total,
            'token_memory_bytes': token_total,
            'memory_efficiency_ratio': memory_ratio,
            'urcm_components': {
                'phoneme_set': phoneme_memory,
                'frequency_vectors': freq_vector_memory,
                'latent_space': latent_memory
            },
            'token_components': {
                'vocabulary': token_vocab_memory,
                'sequence': token_sequence_memory
            },
            'phoneme_set_size': phoneme_set.size,
            'token_vocab_size': vocab_size
        }

        self.results['memory_efficiency'] = results
        return results

    def benchmark_processing_speed(
        self,
        phoneme_set: OptimizedPhonemeSet,
        num_phonemes: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark phoneme processing speed.

        Args:
            phoneme_set: Optimized phoneme set
            num_phonemes: Number of phonemes to process

        Returns:
            Processing speed metrics
        """
        # Get sample phonemes
        sample_phonemes = [
            phoneme_set.get_phoneme(i % phoneme_set.size)
            for i in range(num_phonemes)
        ]

        # Benchmark with cache
        # Pre-warm cache to ensure cached_time reflects retrieval, not generation
        for phoneme in sample_phonemes:
            phoneme_set.get_frequency_vector(phoneme, use_cache=True)
        start_time = time.perf_counter()
        for phoneme in sample_phonemes:
            _ = phoneme_set.get_frequency_vector(phoneme, use_cache=True)
        cached_time = (time.perf_counter() - start_time) * 1000  # ms

        # Clear cache
        phoneme_set._frequency_cache.clear()

        # Benchmark without cache
        start_time = time.perf_counter()
        for phoneme in sample_phonemes:
            phoneme_set.get_frequency_vector(phoneme, use_cache=False)
        uncached_time = (time.perf_counter() - start_time) * 1000  # ms

        results = {
            'cached_time_ms': cached_time,
            'uncached_time_ms': uncached_time,
            'speedup_factor': uncached_time / cached_time if cached_time > 0 else 0,
            'phonemes_processed': num_phonemes,
            'avg_time_per_phoneme_ms': cached_time / num_phonemes if num_phonemes > 0 else 0
        }

        self.results['processing_speed'] = results
        return results

    def benchmark_compression_efficiency(
        self,
        input_dims: List[int],
        compressed_dims: List[int]
    ) -> Dict[str, Any]:
        """
        Benchmark compression efficiency.

        Args:
            input_dims: List of input dimensionalities
            compressed_dims: List of compressed dimensionalities

        Returns:
            Compression efficiency metrics
        """
        compression_ratios = []

        for input_dim, compressed_dim in zip(input_dims, compressed_dims):
            ratio = input_dim / compressed_dim if compressed_dim > 0 else 0
            compression_ratios.append(ratio)

        results = {
            'compression_ratios': compression_ratios,
            'average_ratio': np.mean(compression_ratios),
            'min_ratio': np.min(compression_ratios),
            'max_ratio': np.max(compression_ratios),
            'meets_requirement': np.mean(compression_ratios) >= 2.0  # REQ 10.3
        }

        self.results['compression_efficiency'] = results
        return results

    def benchmark_encoder_latency_comparison(
        self,
        input_dim: int = 24,
        resonance_dim: int = 64,
        seq_len: int = 64,
        runs: int = 50
    ) -> List[Dict[str, Any]]:
        import time

        from urcm.core.resonance_encoder import ResonancePathEncoder
        vectors = np.random.normal(0, 1, (seq_len, input_dim))
        enc_recurrent = ResonancePathEncoder(input_dim=input_dim, resonance_dim=resonance_dim, encoder_type='recurrent_numpy', fast_mode=True)
        enc_transformer = ResonancePathEncoder(input_dim=input_dim, resonance_dim=resonance_dim, encoder_type='transformer_stub')
        start = time.perf_counter()
        for _ in range(runs):
            _ = enc_recurrent.encode_path(vectors)
        t_recurrent = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        for _ in range(runs):
            _ = enc_transformer.encode_path(vectors)
        t_transformer = (time.perf_counter() - start) * 1000.0
        res = [
            {'encoder': 'URCM Recurrent', 'latency_ms': t_recurrent / runs, 'throughput_ops_s': runs / (t_recurrent / 1000.0)},
            {'encoder': 'Transformer Stub', 'latency_ms': t_transformer / runs, 'throughput_ops_s': runs / (t_transformer / 1000.0)},
        ]
        return res

    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        report = ["=" * 80]
        report.append("URCM PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")

        # Memory Efficiency
        if 'memory_efficiency' in self.results:
            mem = self.results['memory_efficiency']
            report.append("MEMORY EFFICIENCY")
            report.append("-" * 80)
            report.append(f"URCM Total Memory:     {mem['urcm_memory_bytes']:,} bytes")
            report.append(f"Token System Memory:   {mem['token_memory_bytes']:,} bytes")
            report.append(f"Efficiency Ratio:      {mem['memory_efficiency_ratio']:.2f}x")
            report.append(f"Phoneme Set Size:      {mem['phoneme_set_size']} phonemes")
            report.append(f"Token Vocab Size:      {mem['token_vocab_size']:,} tokens")
            report.append("")

        # Processing Speed
        if 'processing_speed' in self.results:
            speed = self.results['processing_speed']
            report.append("PROCESSING SPEED")
            report.append("-" * 80)
            report.append(f"Cached Processing:     {speed['cached_time_ms']:.3f} ms")
            report.append(f"Uncached Processing:   {speed['uncached_time_ms']:.3f} ms")
            report.append(f"Cache Speedup:         {speed['speedup_factor']:.2f}x")
            report.append(f"Avg Time/Phoneme:      {speed['avg_time_per_phoneme_ms']:.4f} ms")
            report.append("")

        # Compression Efficiency
        if 'compression_efficiency' in self.results:
            comp = self.results['compression_efficiency']
            report.append("COMPRESSION EFFICIENCY")
            report.append("-" * 80)
            report.append(f"Average Ratio:         {comp['average_ratio']:.2f}x")
            report.append(f"Min Ratio:             {comp['min_ratio']:.2f}x")
            report.append(f"Max Ratio:             {comp['max_ratio']:.2f}x")
            report.append(f"Meets Requirement:     {comp['meets_requirement']}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

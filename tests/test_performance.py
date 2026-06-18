"""
Property tests for computational efficiency constraints.

Tests Property 11: Computational Efficiency Constraints
Validates Requirements: 10.1, 10.2, 10.3, 10.5
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from urcm.core.performance import (
    OptimizedPhonemeSet,
    CompressionMonitor,
    PerformanceBenchmark,
    PerformanceMetrics
)


class TestComputationalEfficiency:
    """
    Property 11: Computational Efficiency Constraints
    
    Validates that URCM meets all performance requirements:
    - REQ 10.1: Small finite phoneme set
    - REQ 10.2: K-dimensional frequency processing (K ∈ [16, 32])
    - REQ 10.3: Compression efficiency
    - REQ 10.5: Memory efficiency vs token-based systems
    """
    
    def test_property_phoneme_set_size_constraint(self):
        """
        REQ 10.1: System SHALL use a small finite phoneme set.
        
        Validates that phoneme set is significantly smaller than token vocabularies.
        """
        phoneme_set = OptimizedPhonemeSet()
        
        # Property: Phoneme set must be finite and small
        assert phoneme_set.size > 0, "Phoneme set must not be empty"
        assert phoneme_set.size < 100, "Phoneme set must be small (< 100 phonemes)"
        
        # Compare to typical token vocabularies (30k-50k)
        typical_token_vocab_size = 50000
        size_ratio = typical_token_vocab_size / phoneme_set.size
        
        assert size_ratio > 500, f"Phoneme set must be much smaller than token vocab (ratio: {size_ratio})"
        
        # Verify all phonemes are accessible
        for i in range(phoneme_set.size):
            phoneme = phoneme_set.get_phoneme(i)
            assert phoneme is not None, f"Phoneme at index {i} must be accessible"
            
            # Verify bidirectional mapping
            phoneme_id = phoneme_set.get_phoneme_id(phoneme)
            assert phoneme_id == i, "Phoneme ID mapping must be consistent"
    
    @given(st.integers(min_value=16, max_value=32))
    @settings(max_examples=10, deadline=1000)  # Allow up to 1000ms to prevent flaky failures
    def test_property_frequency_dimension_constraint(self, K: int):
        """
        REQ 10.2: Frequency processing SHALL operate in K-dimensional space where K ∈ [16, 32].
        
        Property: All frequency vectors must have dimensionality K ∈ [16, 32].
        """
        phoneme_set = OptimizedPhonemeSet(vector_dimension=K)
        
        # Test multiple phonemes
        for i in range(min(10, phoneme_set.size)):
            phoneme = phoneme_set.get_phoneme(i)
            freq_vector = phoneme_set.get_frequency_vector(phoneme)
            
            # Property: Dimension must be exactly K
            assert freq_vector.shape == (K,), f"Frequency vector must have dimension {K}"
            
            # Property: Values must be finite and in valid range
            assert np.all(np.isfinite(freq_vector)), "All frequency values must be finite"
            assert np.all(freq_vector >= 0), "Frequency values must be non-negative"
            assert np.all(freq_vector <= 1), "Frequency values must be normalized"
    
    def test_property_memory_efficiency_dtype(self):
        """
        REQ 10.5: Memory efficiency through compact data types.
        
        Property: Frequency vectors use float32 (not float64) for memory efficiency.
        """
        phoneme_set = OptimizedPhonemeSet()
        
        for i in range(min(5, phoneme_set.size)):
            phoneme = phoneme_set.get_phoneme(i)
            freq_vector = phoneme_set.get_frequency_vector(phoneme)
            
            # Property: Must use float32 for memory efficiency
            assert freq_vector.dtype == np.float32, "Frequency vectors must use float32"
            
            # Calculate memory savings
            float32_size = freq_vector.nbytes
            float64_size = freq_vector.shape[0] * 8
            memory_savings = (float64_size - float32_size) / float64_size
            
            assert abs(memory_savings - 0.5) < 1e-9, f"float32 should save 50% memory vs float64, got {memory_savings}"
    
    @given(st.lists(st.integers(min_value=64, max_value=512), min_size=5, max_size=10))
    @settings(max_examples=10)
    def test_property_compression_efficiency(self, input_dims: list):
        """
        REQ 10.3: System SHALL achieve compression efficiency through semantic latent space.
        
        Property: Compression ratio must be >= 2.0 for efficiency.
        """
        monitor = CompressionMonitor()
        
        # Simulate compression operations
        for input_dim in input_dims:
            # Typical compression to latent space (e.g., 32 dim for input >= 64)
            compressed_dim = 32
            
            # Simulate compression
            monitor.record_compression(input_dim, compressed_dim)
        
        # Property: Average compression ratio must meet threshold
        avg_ratio = monitor.get_average_compression_ratio()
        assert avg_ratio >= 2.0, f"Compression ratio {avg_ratio} must be >= 2.0"
        
        # Property: Efficiency validation must pass
        assert monitor.validate_efficiency(), "Compression efficiency validation must pass"
        
        # Property: All individual compressions should be efficient
        efficiency = monitor.get_compression_efficiency()
        assert efficiency['min_ratio'] >= 1.0, "All compressions must reduce size"
        assert efficiency['meets_threshold'], "Must meet efficiency threshold"
    
    def test_property_cache_performance(self):
        """
        REQ 10.5: Memory efficiency through caching.
        
        Property: Cache must improve performance without excessive memory use.
        """
        phoneme_set = OptimizedPhonemeSet(max_cache_size=50)
        
        # Access same phonemes multiple times
        test_phonemes = [phoneme_set.get_phoneme(i % 10) for i in range(100)]
        
        for phoneme in test_phonemes:
            phoneme_set.get_frequency_vector(phoneme, use_cache=True)
        
        # Property: Cache hit rate should be high for repeated access
        stats = phoneme_set.get_cache_stats()
        assert stats['cache_hits'] > 0, "Cache must register hits"
        assert stats['hit_rate'] > 0.5, "Cache hit rate should be > 50% for repeated access"
        
        # Property: Cache size must respect limit
        assert stats['cache_size'] <= stats['max_cache_size'], "Cache must not exceed max size"
        
        # Property: Memory usage must be reasonable
        memory_usage = phoneme_set.get_memory_usage()
        max_expected_memory = stats['max_cache_size'] * phoneme_set.vector_dimension * 4 * 2
        assert memory_usage < max_expected_memory, "Memory usage must be within expected bounds"
    
    def test_property_memory_efficiency_vs_tokens(self):
        """
        REQ 10.5: System SHALL demonstrate memory efficiency compared to token-based systems.
        
        Property: URCM must use less memory than token-based systems for equivalent processing.
        """
        benchmark = PerformanceBenchmark()
        phoneme_set = OptimizedPhonemeSet()
        
        # Test with various text lengths
        test_texts = [
            "Short text",
            "Medium length text with more words and complexity",
            "A much longer text that simulates a paragraph of reasonable length with multiple sentences and ideas."
        ]
        
        for text in test_texts:
            results = benchmark.benchmark_memory_efficiency(
                text=text,
                phoneme_set=phoneme_set,
                latent_dim=128
            )
            
            # Property: URCM must use less memory
            assert results['urcm_memory_bytes'] < results['token_memory_bytes'], \
                "URCM must use less memory than token-based systems"
            
            # Property: Efficiency ratio must be > 1 (token system uses more)
            assert results['memory_efficiency_ratio'] > 1.0, \
                f"Memory efficiency ratio {results['memory_efficiency_ratio']} must be > 1.0"
            
            # Property: Phoneme set must be much smaller than token vocab
            assert results['phoneme_set_size'] < results['token_vocab_size'] / 100, \
                "Phoneme set must be at least 100x smaller than token vocabulary"
    
    def test_property_processing_speed_scalability(self):
        """
        REQ 10.4: Scalable processing through efficient operations.
        
        Property: Processing speed must scale linearly with input size.
        """
        benchmark = PerformanceBenchmark()
        phoneme_set = OptimizedPhonemeSet()
        
        # Test different input sizes
        sizes = [10, 50, 100]
        times = []
        
        for size in sizes:
            results = benchmark.benchmark_processing_speed(
                phoneme_set=phoneme_set,
                num_phonemes=size
            )
            times.append(results['cached_time_ms'])
        
        # Property: Time should scale roughly linearly
        # Check that doubling input doesn't more than triple time (allowing overhead)
        for i in range(len(sizes) - 1):
            size_ratio = sizes[i + 1] / sizes[i]
            time_ratio = times[i + 1] / times[i] if times[i] > 0 else 0
            
            # Allow some overhead but ensure sub-quadratic scaling
            assert time_ratio < size_ratio * 2.5, \
                f"Processing time must scale sub-quadratically (size ratio: {size_ratio}, time ratio: {time_ratio})"
    
    def test_property_deterministic_frequency_generation(self):
        """
        REQ 10.2: Frequency vectors must be deterministic for reproducibility.
        
        Property: Same phoneme always produces same frequency vector.
        """
        phoneme_set1 = OptimizedPhonemeSet(vector_dimension=24)
        phoneme_set2 = OptimizedPhonemeSet(vector_dimension=24)
        
        # Test multiple phonemes
        for i in range(min(10, phoneme_set1.size)):
            phoneme = phoneme_set1.get_phoneme(i)
            
            vec1 = phoneme_set1.get_frequency_vector(phoneme, use_cache=False)
            vec2 = phoneme_set2.get_frequency_vector(phoneme, use_cache=False)
            
            # Property: Vectors must be identical
            assert np.allclose(vec1, vec2), \
                f"Frequency vectors for '{phoneme}' must be deterministic"
    
    def test_property_compression_ratio_bounds(self):
        """
        REQ 10.3: Compression must be efficient but not lossy beyond recovery.
        
        Property: Compression ratio must be in reasonable bounds [2.0, 10.0].
        """
        monitor = CompressionMonitor()
        
        # Test various compression scenarios
        test_cases = [
            (256, 128),  # 2x compression
            (512, 128),  # 4x compression
            (1024, 128), # 8x compression
        ]
        
        for input_dim, compressed_dim in test_cases:
            monitor.record_compression(input_dim, compressed_dim)
        
        efficiency = monitor.get_compression_efficiency()
        
        # Property: Compression must be efficient
        assert efficiency['min_ratio'] >= 2.0, "Minimum compression ratio must be >= 2.0"
        
        # Property: Compression must not be excessive (information preservation)
        assert efficiency['max_ratio'] <= 10.0, "Maximum compression ratio should be <= 10.0"
        
        # Property: Average should be in sweet spot
        assert 2.0 <= efficiency['average_ratio'] <= 8.0, \
            "Average compression ratio should be in [2.0, 8.0] range"
    
    def test_property_phoneme_set_completeness(self):
        """
        REQ 10.1: Phoneme set must provide complete articulatory coverage.
        
        Property: Phoneme set must cover all major phonetic categories.
        """
        phoneme_set = OptimizedPhonemeSet()
        
        # Get all phonemes
        all_phonemes = [phoneme_set.get_phoneme(i) for i in range(phoneme_set.size)]
        
        # Property: Must have vowels
        vowels = ['a', 'i', 'u', 'e', 'o']
        vowel_coverage = sum(1 for v in vowels if v in all_phonemes)
        assert vowel_coverage >= 3, "Must have at least 3 basic vowels"
        
        # Property: Must have stops
        stops = ['k', 'g', 't', 'd', 'p', 'b']
        stop_coverage = sum(1 for s in stops if s in all_phonemes)
        assert stop_coverage >= 4, "Must have at least 4 stop consonants"
        
        # Property: Must have fricatives
        fricatives = ['s', 'h', 'ś', 'ṣ']
        fricative_coverage = sum(1 for f in fricatives if f in all_phonemes)
        assert fricative_coverage >= 2, "Must have at least 2 fricatives"
        
        # Property: Total coverage must be comprehensive but finite
        assert len(all_phonemes) >= 40, "Must have at least 40 phonemes for coverage"
        assert len(all_phonemes) <= 80, "Must not exceed 80 phonemes (finite constraint)"


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""
    
    def test_compression_ratio_calculation(self):
        """Test automatic compression ratio calculation."""
        metrics = PerformanceMetrics(
            input_size_bytes=1000,
            compressed_size_bytes=250
        )
        
        assert metrics.compression_ratio == 4.0, "Compression ratio should be 4.0"
    
    def test_memory_efficiency_ratio_calculation(self):
        """Test automatic memory efficiency ratio calculation."""
        metrics = PerformanceMetrics(
            total_memory_bytes=1000,
            token_memory_bytes=5000
        )
        
        assert metrics.memory_efficiency_ratio == 5.0, "Memory efficiency ratio should be 5.0"
    
    def test_zero_division_handling(self):
        """Test that zero values don't cause division errors."""
        metrics = PerformanceMetrics(
            input_size_bytes=1000,
            compressed_size_bytes=0
        )
        
        # Should not raise error
        assert metrics.compression_ratio == 0.0


class TestBenchmarkReporting:
    """Tests for benchmark report generation."""
    
    def test_comprehensive_benchmark_report(self):
        """Test that benchmark generates comprehensive report."""
        benchmark = PerformanceBenchmark()
        phoneme_set = OptimizedPhonemeSet()
        
        # Run all benchmarks
        benchmark.benchmark_memory_efficiency("Test text", phoneme_set)
        benchmark.benchmark_processing_speed(phoneme_set, num_phonemes=50)
        benchmark.benchmark_compression_efficiency([256, 512], [128, 128])
        
        # Generate report
        report = benchmark.generate_report()
        
        # Verify report contains all sections
        assert "MEMORY EFFICIENCY" in report
        assert "PROCESSING SPEED" in report
        assert "COMPRESSION EFFICIENCY" in report
        
        # Verify key metrics are present
        assert "Efficiency Ratio:" in report
        assert "Cache Speedup:" in report
        assert "Average Ratio:" in report

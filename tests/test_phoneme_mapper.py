"""
Property-based tests for the phoneme-frequency mapping system.

This module contains property-based tests that validate the phoneme-to-frequency
mapping functionality across all possible valid inputs.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.extra.numpy import arrays

from urcm.core.phoneme_mapper import (
    PhonemeFrequencyMapper,
    TextToPhonemeConverter,
    PhonemeFrequencyPipeline
)
from urcm.core.data_models import PhonemeSequence, FrequencyPath
from urcm.core.validation import DataValidation


# Hypothesis strategies for generating test data
@st.composite
def valid_phoneme_strategy(draw):
    """Generate valid Sanskrit phonemes from the mapper's phoneme set."""
    mapper = PhonemeFrequencyMapper()
    phoneme = draw(st.sampled_from(list(mapper.phoneme_space)))
    return phoneme


@st.composite
def valid_phoneme_sequence_strategy(draw):
    """Generate valid sequences of Sanskrit phonemes."""
    mapper = PhonemeFrequencyMapper()
    phonemes = draw(st.lists(
        st.sampled_from(list(mapper.phoneme_space)),
        min_size=1,
        max_size=10
    ))
    return phonemes


@st.composite
def frequency_dimension_strategy(draw):
    """Generate valid frequency dimensions K ∈ [16, 32]."""
    return draw(st.integers(min_value=16, max_value=32))


@st.composite
def text_input_strategy(draw):
    """Generate valid text inputs for the pipeline."""
    text = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz ",
        min_size=1,
        max_size=50
    ))
    # Ensure text is not just whitespace
    assume(text.strip())
    return text


class TestPhonemeFrequencyMappingConsistency:
    """Property-based tests for phoneme-frequency mapping consistency."""
    
    @given(valid_phoneme_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    @pytest.mark.property
    def test_phoneme_to_frequency_mapping_consistency(self, phoneme, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 1: Phoneme-to-Frequency Mapping Consistency
        
        For any valid phoneme from the Sanskrit-derived phoneme set, the frequency mapper
        should produce a vector in K-dimensional space where K ∈ [16, 32], and the mapping
        should be deterministic and consistent across all invocations.
        
        **Validates: Requirements 1.2**
        """
        # Create mapper with specified frequency dimension
        mapper = PhonemeFrequencyMapper(frequency_dim=frequency_dim)
        
        # Map the phoneme multiple times
        vector1 = mapper.map_phoneme(phoneme)
        vector2 = mapper.map_phoneme(phoneme)
        vector3 = mapper.map_phoneme(phoneme)
        
        # Check dimensionality constraints
        assert vector1.shape == (frequency_dim,), f"Vector should have shape ({frequency_dim},), got {vector1.shape}"
        assert vector2.shape == (frequency_dim,), f"Vector should have shape ({frequency_dim},), got {vector2.shape}"
        assert vector3.shape == (frequency_dim,), f"Vector should have shape ({frequency_dim},), got {vector3.shape}"
        
        # Check that frequency dimension is in valid range [16, 32]
        assert 16 <= frequency_dim <= 32, f"Frequency dimension {frequency_dim} not in range [16, 32]"
        
        # Check deterministic mapping - same phoneme should always produce same vector
        np.testing.assert_array_equal(vector1, vector2, 
                                    err_msg=f"Mapping for phoneme '{phoneme}' is not deterministic")
        np.testing.assert_array_equal(vector2, vector3,
                                    err_msg=f"Mapping for phoneme '{phoneme}' is not deterministic")
        
        # Check that vectors are finite (no NaN or infinity)
        assert np.all(np.isfinite(vector1)), f"Vector for phoneme '{phoneme}' contains non-finite values"
        
        # Check that vectors are normalized (unit vectors)
        vector_norm = np.linalg.norm(vector1)
        assert abs(vector_norm - 1.0) < 1e-6, f"Vector for phoneme '{phoneme}' is not normalized: norm={vector_norm}"
        
        # Check that the phoneme is actually in the mapper's phoneme space
        assert phoneme in mapper.phoneme_space, f"Phoneme '{phoneme}' not in mapper's phoneme space"
    
    @given(valid_phoneme_sequence_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    @pytest.mark.property
    def test_phoneme_sequence_mapping_consistency(self, phonemes, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 1: Phoneme-to-Frequency Mapping Consistency
        
        For any valid sequence of phonemes, the mapper should produce a frequency path
        with correct dimensionality and consistent individual phoneme mappings.
        
        **Validates: Requirements 1.2**
        """
        # Create mapper with specified frequency dimension
        mapper = PhonemeFrequencyMapper(frequency_dim=frequency_dim)
        
        # Map the sequence
        frequency_path = mapper.map_sequence(phonemes)
        
        # Check overall shape
        expected_shape = (len(phonemes), frequency_dim)
        assert frequency_path.shape == expected_shape, f"Expected shape {expected_shape}, got {frequency_path.shape}"
        
        # Check that each individual phoneme mapping is consistent
        for i, phoneme in enumerate(phonemes):
            individual_vector = mapper.map_phoneme(phoneme)
            sequence_vector = frequency_path[i]
            
            # The vectors should be close (allowing for smoothing effects)
            # We check that they're in the same general direction
            dot_product = np.dot(individual_vector, sequence_vector)
            individual_norm = np.linalg.norm(individual_vector)
            sequence_norm = np.linalg.norm(sequence_vector)
            
            if individual_norm > 0 and sequence_norm > 0:
                cosine_similarity = dot_product / (individual_norm * sequence_norm)
                # Allow for some variation due to smoothing, but should be generally similar
                assert cosine_similarity > 0.5, f"Phoneme '{phoneme}' mapping inconsistent in sequence at position {i}"
        
        # Check that all vectors are finite
        assert np.all(np.isfinite(frequency_path)), "Frequency path contains non-finite values"
    
    @given(valid_phoneme_sequence_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    @pytest.mark.property
    def test_frequency_path_creation_consistency(self, phonemes, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 1: Phoneme-to-Frequency Mapping Consistency
        
        For any valid phoneme sequence, the create_frequency_path method should produce
        a valid FrequencyPath object that passes validation.
        
        **Validates: Requirements 1.2**
        """
        # Create mapper with specified frequency dimension
        mapper = PhonemeFrequencyMapper(frequency_dim=frequency_dim)
        
        # Create frequency path
        freq_path = mapper.create_frequency_path(phonemes)
        
        # Check that it's a valid FrequencyPath object
        assert isinstance(freq_path, FrequencyPath), "Output should be a FrequencyPath object"
        
        # Check that it passes validation
        assert DataValidation.validate_frequency_path(freq_path), "FrequencyPath should pass validation"
        
        # Check dimensionality
        assert freq_path.vectors.shape == (len(phonemes), frequency_dim), "Incorrect vector dimensions"
        
        # Check phoneme mapping consistency
        assert len(freq_path.phoneme_mapping) == len(phonemes), "Phoneme mapping length mismatch"
        
        for i, (mapped_phoneme, index) in enumerate(freq_path.phoneme_mapping):
            assert mapped_phoneme == phonemes[i], f"Phoneme mapping mismatch at position {i}"
            assert index == i, f"Index mapping mismatch at position {i}"
        
        # Check smoothness score is non-negative
        assert freq_path.smoothness_score >= 0, "Smoothness score should be non-negative"
        
        # Check that smoothness score is finite
        assert np.isfinite(freq_path.smoothness_score), "Smoothness score should be finite"
    
    @given(st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=5))
    @settings(max_examples=50)
    @pytest.mark.property
    def test_mapper_initialization_consistency(self, freq_dim_offset, smoothness_scale):
        """
        Feature: urcm-reasoning-system, Property 1: Phoneme-to-Frequency Mapping Consistency
        
        For any valid initialization parameters, the mapper should initialize consistently
        and maintain the same phoneme set and properties.
        
        **Validates: Requirements 1.2**
        """
        # Generate valid parameters
        frequency_dim = 16 + freq_dim_offset  # Ensures K ∈ [17, 21] for this test
        smoothness_weight = smoothness_scale * 0.02  # Ensures reasonable smoothness weights
        
        # Create two mappers with same parameters
        mapper1 = PhonemeFrequencyMapper(frequency_dim=frequency_dim, smoothness_weight=smoothness_weight)
        mapper2 = PhonemeFrequencyMapper(frequency_dim=frequency_dim, smoothness_weight=smoothness_weight)
        
        # Check that they have the same properties
        assert mapper1.K == mapper2.K == frequency_dim, "Frequency dimensions should match"
        assert mapper1.smoothness_weight == mapper2.smoothness_weight == smoothness_weight, "Smoothness weights should match"
        assert mapper1.phoneme_space == mapper2.phoneme_space, "Phoneme spaces should be identical"
        
        # Check that they produce the same mappings (deterministic initialization)
        test_phoneme = 'a'  # Use a common phoneme
        vector1 = mapper1.map_phoneme(test_phoneme)
        vector2 = mapper2.map_phoneme(test_phoneme)
        
        np.testing.assert_array_equal(vector1, vector2, 
                                    err_msg="Mappers with same parameters should produce identical mappings")


class TestFrequencyPathSmoothness:
    """Property-based tests for frequency path smoothness constraints."""
    
    @given(valid_phoneme_sequence_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    @pytest.mark.property
    def test_frequency_path_smoothness_constraints(self, phonemes, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 2: Frequency Path Smoothness
        
        For any sequence of adjacent phonemes, the resulting frequency vectors should
        satisfy smoothness constraints such that ||f(pi) - f(pj)||² is minimized,
        ensuring natural transitions in frequency space.
        
        **Validates: Requirements 1.3**
        """
        # Skip single phoneme sequences (no smoothness to test)
        assume(len(phonemes) > 1)
        
        # Create mapper
        mapper = PhonemeFrequencyMapper(frequency_dim=frequency_dim, smoothness_weight=0.1)
        
        # Get original (unsmoothed) path by mapping individual phonemes
        original_vectors = np.array([mapper.map_phoneme(p) for p in phonemes])
        
        # Get smoothed path through sequence mapping
        smoothed_path = mapper.map_sequence(phonemes)
        
        # Calculate smoothness metrics
        original_smoothness = mapper.calculate_smoothness_score(original_vectors)
        smoothed_smoothness = mapper.calculate_smoothness_score(smoothed_path)
        
        # Smoothed path should be at least as smooth as original (lower or equal score)
        # We allow a small margin for edge cases where local constraints might slightly increase global score
        assert smoothed_smoothness <= original_smoothness + 0.5, \
            f"Smoothed path should be smoother: original={original_smoothness:.6f}, smoothed={smoothed_smoothness:.6f}"
        
        # Check that adjacent vectors in smoothed path are reasonably close
        if len(phonemes) > 1:
            diffs = np.diff(smoothed_path, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            
            # All distances should be finite and reasonable
            assert np.all(np.isfinite(distances)), "All inter-vector distances should be finite"
            
            # No distance should be excessively large (indicating poor smoothness)
            max_distance = np.max(distances)
            assert max_distance < 5.0, f"Maximum distance {max_distance} too large, indicates poor smoothness"
            
            # The smoothness score should reflect the actual path smoothness
            expected_smoothness = np.mean(distances)
            assert abs(smoothed_smoothness - expected_smoothness) < 1e-6, \
                f"Smoothness score {smoothed_smoothness} should match calculated smoothness {expected_smoothness}"
    
    @given(valid_phoneme_sequence_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    @pytest.mark.property
    def test_smoothness_enforcement_effectiveness(self, phonemes, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 2: Frequency Path Smoothness
        
        For any phoneme sequence, the smoothness enforcement should create more
        natural transitions while preserving the overall semantic structure.
        
        **Validates: Requirements 1.3**
        """
        # Skip single phoneme sequences
        assume(len(phonemes) > 1)
        
        # Test with different smoothness weights
        weak_smoother = PhonemeFrequencyMapper(frequency_dim=frequency_dim, smoothness_weight=0.01)
        strong_smoother = PhonemeFrequencyMapper(frequency_dim=frequency_dim, smoothness_weight=0.3)
        
        weak_path = weak_smoother.map_sequence(phonemes)
        strong_path = strong_smoother.map_sequence(phonemes)
        
        weak_smoothness = weak_smoother.calculate_smoothness_score(weak_path)
        strong_smoothness = strong_smoother.calculate_smoothness_score(strong_path)
        
        # Stronger smoothing should generally produce smoother paths
        # (allowing some tolerance for edge cases)
        if len(phonemes) > 2:  # Only test for longer sequences where smoothing has more effect
            assert strong_smoothness <= weak_smoothness + 0.1, \
                f"Strong smoothing should produce smoother paths: weak={weak_smoothness:.6f}, strong={strong_smoothness:.6f}"
        
        # Both paths should maintain the same sequence length and dimensionality
        assert weak_path.shape == strong_path.shape == (len(phonemes), frequency_dim)
        
        # Both paths should be valid
        assert np.all(np.isfinite(weak_path)), "Weakly smoothed path should be finite"
        assert np.all(np.isfinite(strong_path)), "Strongly smoothed path should be finite"
    
    @given(st.lists(st.sampled_from(['a', 'i', 'u']), min_size=3, max_size=8), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    @pytest.mark.property
    def test_smoothness_with_similar_phonemes(self, phonemes, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 2: Frequency Path Smoothness
        
        For sequences of similar phonemes (vowels), the smoothness constraints
        should produce very smooth transitions due to articulatory similarity.
        
        **Validates: Requirements 1.3**
        """
        # Use vowels which should be naturally smooth
        mapper = PhonemeFrequencyMapper(frequency_dim=frequency_dim)
        
        smoothed_path = mapper.map_sequence(phonemes)
        smoothness_score = mapper.calculate_smoothness_score(smoothed_path)
        
        # Similar phonemes should produce relatively smooth paths
        # (vowels should have lower smoothness scores than mixed consonant-vowel sequences)
        assert smoothness_score < 3.0, f"Similar phonemes should produce smooth paths, got score: {smoothness_score}"
        
        # Check that the path maintains reasonable vector magnitudes
        norms = np.linalg.norm(smoothed_path, axis=1)
        assert np.all(norms > 0.5), "All vectors should have reasonable magnitude"
        assert np.all(norms < 2.0), "No vector should have excessive magnitude"
    
    @given(valid_phoneme_sequence_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    @pytest.mark.property
    def test_smoothness_preserves_phoneme_identity(self, phonemes, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 2: Frequency Path Smoothness
        
        For any phoneme sequence, smoothness enforcement should preserve the
        essential identity of each phoneme while improving transitions.
        
        **Validates: Requirements 1.3**
        """
        mapper = PhonemeFrequencyMapper(frequency_dim=frequency_dim)
        
        # Get individual phoneme vectors and smoothed sequence
        individual_vectors = [mapper.map_phoneme(p) for p in phonemes]
        smoothed_path = mapper.map_sequence(phonemes)
        
        # Check that smoothed vectors are still reasonably similar to originals
        for i, (original, smoothed) in enumerate(zip(individual_vectors, smoothed_path)):
            # Calculate cosine similarity
            dot_product = np.dot(original, smoothed)
            original_norm = np.linalg.norm(original)
            smoothed_norm = np.linalg.norm(smoothed)
            
            if original_norm > 0 and smoothed_norm > 0:
                cosine_similarity = dot_product / (original_norm * smoothed_norm)
                
                # Smoothed vector should still be reasonably similar to original
                # (allowing for smoothing effects, especially at sequence boundaries)
                min_similarity = 0.3 if len(phonemes) > 3 else 0.5
                assert cosine_similarity > min_similarity, \
                    f"Phoneme '{phonemes[i]}' at position {i} lost too much identity: similarity={cosine_similarity:.3f}"


class TestTextToFrequencyPipelineCompleteness:
    """Property-based tests for complete text-to-frequency pipeline."""
    
    @given(text_input_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100, deadline=None)
    @pytest.mark.property
    def test_text_to_frequency_pipeline_completeness(self, text, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 3: Text-to-Frequency Pipeline Completeness
        
        For any valid text input, the complete processing pipeline (text → phoneme sequence
        → frequency path) should produce a valid frequency path with proper dimensionality
        and smoothness properties.
        
        **Validates: Requirements 1.4**
        """
        # Create pipeline
        pipeline = PhonemeFrequencyPipeline(frequency_dim=frequency_dim)
        
        # Process text through complete pipeline
        freq_path = pipeline.process_text(text)
        
        # Check that output is a valid FrequencyPath
        assert isinstance(freq_path, FrequencyPath), "Pipeline should produce FrequencyPath object"
        
        # Check that it passes validation
        assert DataValidation.validate_frequency_path(freq_path), "Pipeline output should pass validation"
        
        # Check dimensionality constraints
        assert freq_path.vectors.shape[1] == frequency_dim, f"Output should have {frequency_dim} dimensions"
        assert 16 <= frequency_dim <= 32, "Frequency dimension should be in valid range [16, 32]"
        
        # Check that sequence length is reasonable for input text
        # (should be at least 1, and generally proportional to text length)
        assert freq_path.vectors.shape[0] >= 1, "Should produce at least one frequency vector"
        assert freq_path.vectors.shape[0] <= len(text) + 5, "Sequence length should be reasonable for input text"
        
        # Check that all vectors are finite
        assert np.all(np.isfinite(freq_path.vectors)), "All frequency vectors should be finite"
        
        # Check smoothness score properties
        assert freq_path.smoothness_score >= 0, "Smoothness score should be non-negative"
        assert np.isfinite(freq_path.smoothness_score), "Smoothness score should be finite"
        
        # Check phoneme mapping consistency
        assert len(freq_path.phoneme_mapping) == freq_path.vectors.shape[0], "Phoneme mapping should match vector count"
        
        for phoneme, index in freq_path.phoneme_mapping:
            assert isinstance(phoneme, str), "Phoneme should be a string"
            assert isinstance(index, int), "Index should be an integer"
            assert 0 <= index < freq_path.vectors.shape[0], "Index should be valid"
        
        # Check that pipeline validation agrees with our assessment
        assert pipeline.validate_pipeline(text), "Pipeline validation should pass for valid text"
    
    @given(text_input_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    @pytest.mark.property
    def test_pipeline_deterministic_behavior(self, text, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 3: Text-to-Frequency Pipeline Completeness
        
        For any valid text input, the pipeline should produce identical results
        when run multiple times (deterministic behavior).
        
        **Validates: Requirements 1.4**
        """
        # Create pipeline
        pipeline = PhonemeFrequencyPipeline(frequency_dim=frequency_dim)
        
        # Process same text multiple times
        result1 = pipeline.process_text(text)
        result2 = pipeline.process_text(text)
        result3 = pipeline.process_text(text)
        
        # Check that all results are identical
        np.testing.assert_array_equal(result1.vectors, result2.vectors,
                                    err_msg=f"Pipeline should be deterministic for text: '{text}'")
        np.testing.assert_array_equal(result2.vectors, result3.vectors,
                                    err_msg=f"Pipeline should be deterministic for text: '{text}'")
        
        assert result1.smoothness_score == result2.smoothness_score == result3.smoothness_score, \
            "Smoothness scores should be identical"
        
        assert result1.phoneme_mapping == result2.phoneme_mapping == result3.phoneme_mapping, \
            "Phoneme mappings should be identical"
    
    @given(text_input_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    @pytest.mark.property
    def test_pipeline_with_different_dimensions(self, text):
        """
        Feature: urcm-reasoning-system, Property 3: Text-to-Frequency Pipeline Completeness
        
        For any valid text input, the pipeline should work correctly with different
        frequency dimensions within the valid range [16, 32].
        
        **Validates: Requirements 1.4**
        """
        # Test with different valid dimensions
        dimensions = [16, 20, 24, 28, 32]
        results = []
        
        for dim in dimensions:
            pipeline = PhonemeFrequencyPipeline(frequency_dim=dim)
            freq_path = pipeline.process_text(text)
            results.append(freq_path)
            
            # Check dimension-specific properties
            assert freq_path.vectors.shape[1] == dim, f"Should have {dim} dimensions"
            assert DataValidation.validate_frequency_path(freq_path), f"Should be valid with {dim} dimensions"
        
        # Check that sequence lengths are consistent across dimensions
        # (same text should produce same number of phonemes regardless of dimension)
        sequence_lengths = [result.vectors.shape[0] for result in results]
        assert all(length == sequence_lengths[0] for length in sequence_lengths), \
            "Sequence length should be consistent across dimensions"
        
        # Check that phoneme mappings are consistent
        phoneme_mappings = [result.phoneme_mapping for result in results]
        assert all(mapping == phoneme_mappings[0] for mapping in phoneme_mappings), \
            "Phoneme mappings should be consistent across dimensions"
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=3))
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    @pytest.mark.property
    def test_pipeline_with_simple_inputs(self, simple_text):
        """
        Feature: urcm-reasoning-system, Property 3: Text-to-Frequency Pipeline Completeness
        
        For simple text inputs (single words), the pipeline should produce
        clean, well-formed frequency paths.
        
        **Validates: Requirements 1.4**
        """
        pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
        
        freq_path = pipeline.process_text(simple_text)
        
        # Simple inputs should produce reasonable outputs
        assert 1 <= freq_path.vectors.shape[0] <= len(simple_text) + 2, \
            "Simple text should produce reasonable sequence length"
        
        # All phonemes should be valid
        for phoneme, _ in freq_path.phoneme_mapping:
            assert phoneme in pipeline.frequency_mapper.phoneme_space, \
                f"Phoneme '{phoneme}' should be in valid phoneme space"
        
        # Vectors should have reasonable magnitudes
        norms = np.linalg.norm(freq_path.vectors, axis=1)
        assert np.all(norms > 0.1), "All vectors should have reasonable magnitude"
        assert np.all(norms < 5.0), "No vector should have excessive magnitude"


class TestPhonemeFrequencyPipelineConsistency:
    """Property-based tests for the complete text-to-frequency pipeline."""
    
    @given(text_input_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    @pytest.mark.property
    def test_pipeline_consistency(self, text, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 1: Phoneme-to-Frequency Mapping Consistency
        
        For any valid text input, the complete pipeline should produce consistent
        and valid frequency path outputs.
        
        **Validates: Requirements 1.2**
        """
        # Create pipeline
        pipeline = PhonemeFrequencyPipeline(frequency_dim=frequency_dim)
        
        # Process text multiple times
        freq_path1 = pipeline.process_text(text)
        freq_path2 = pipeline.process_text(text)
        
        # Check that results are identical (deterministic)
        np.testing.assert_array_equal(freq_path1.vectors, freq_path2.vectors,
                                    err_msg=f"Pipeline should be deterministic for text: '{text}'")
        
        assert freq_path1.smoothness_score == freq_path2.smoothness_score, "Smoothness scores should be identical"
        assert freq_path1.phoneme_mapping == freq_path2.phoneme_mapping, "Phoneme mappings should be identical"
        
        # Check that output is valid
        assert DataValidation.validate_frequency_path(freq_path1), "Pipeline output should pass validation"
        
        # Check dimensionality
        assert freq_path1.vectors.shape[1] == frequency_dim, f"Output dimension should be {frequency_dim}"
        
        # Check that pipeline validation works
        assert pipeline.validate_pipeline(text), f"Pipeline validation should pass for text: '{text}'"


class TestPhonemeMapperEdgeCases:
    """Unit tests for edge cases and error conditions."""
    
    @pytest.mark.unit
    def test_invalid_phoneme_rejection(self):
        """Test that invalid phonemes are properly rejected."""
        mapper = PhonemeFrequencyMapper()
        
        with pytest.raises(ValueError, match="not in Sanskrit phoneme set"):
            mapper.map_phoneme("invalid_phoneme_xyz")
    
    @pytest.mark.unit
    def test_empty_sequence_rejection(self):
        """Test that empty phoneme sequences are properly rejected."""
        mapper = PhonemeFrequencyMapper()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            mapper.map_sequence([])
        
        with pytest.raises(ValueError, match="cannot be empty"):
            mapper.create_frequency_path([])
    
    @pytest.mark.unit
    def test_invalid_frequency_dimension_rejection(self):
        """Test that invalid frequency dimensions are properly rejected."""
        with pytest.raises(ValueError, match="must be in range"):
            PhonemeFrequencyMapper(frequency_dim=10)  # Too small
        
        with pytest.raises(ValueError, match="must be in range"):
            PhonemeFrequencyMapper(frequency_dim=50)  # Too large
    
    @pytest.mark.unit
    def test_empty_text_rejection(self):
        """Test that empty text inputs are properly rejected."""
        pipeline = PhonemeFrequencyPipeline()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            pipeline.process_text("")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            pipeline.process_text("   ")  # Only whitespace
    
    @pytest.mark.unit
    def test_single_phoneme_smoothness(self):
        """Test that single phoneme sequences have zero smoothness score."""
        mapper = PhonemeFrequencyMapper()
        
        # Single phoneme should have zero smoothness (no adjacent phonemes)
        single_path = mapper.map_sequence(['a'])
        smoothness = mapper.calculate_smoothness_score(single_path)
        assert smoothness == 0.0, "Single phoneme should have zero smoothness score"
        
        freq_path = mapper.create_frequency_path(['a'])
        assert freq_path.smoothness_score == 0.0, "Single phoneme FrequencyPath should have zero smoothness score"

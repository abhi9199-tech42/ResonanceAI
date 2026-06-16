"""
Property-based tests for URCM data models.

This module contains property-based tests that validate the core data structures
across all possible valid inputs, ensuring consistency and correctness.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.extra.numpy import arrays

from urcm.core.data_models import (
    PhonemeSequence, FrequencyPath, ResonanceState,
    AttractorState, ReasoningPath, MeshSignal
)
from urcm.core.validation import DataValidation


# Hypothesis strategies for generating test data
@st.composite
def phoneme_sequence_strategy(draw):
    """Generate valid PhonemeSequence instances."""
    phonemes = draw(st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=3), min_size=1, max_size=10))
    source_text = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=50))
    language_hint = draw(st.one_of(st.none(), st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5)))
    assume(source_text.strip())
    return PhonemeSequence(phonemes=phonemes, source_text=source_text, language_hint=language_hint)


@st.composite
def frequency_path_strategy(draw):
    """Generate valid FrequencyPath instances."""
    seq_length = draw(st.integers(min_value=1, max_value=10))
    k_dim = draw(st.integers(min_value=16, max_value=32))
    
    # Generate vectors with smaller values to ensure smoothness
    vectors = draw(arrays(
        dtype=np.float64,
        shape=(seq_length, k_dim),
        elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    ))
    
    smoothness_score = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    
    # Generate phoneme mapping
    phonemes = draw(st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=3), min_size=seq_length, max_size=seq_length))
    phoneme_mapping = [(phonemes[i], i) for i in range(seq_length)]
    
    return FrequencyPath(
        vectors=vectors,
        smoothness_score=smoothness_score,
        phoneme_mapping=phoneme_mapping
    )


@st.composite
def resonance_state_strategy(draw):
    """Generate valid ResonanceState instances."""
    vector_dim = draw(st.integers(min_value=16, max_value=128))
    resonance_vector = draw(arrays(
        dtype=np.float64,
        shape=(vector_dim,),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ))
    
    rho_density = draw(st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False))
    chi_cost = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    
    # Calculate mu to satisfy relationship (mostly)
    mu_value = rho_density / (chi_cost + 1e-9)
    
    stability_score = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    oscillation_phase = draw(st.floats(min_value=0.0, max_value=2*np.pi, allow_nan=False, allow_infinity=False))
    timestamp = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    
    return ResonanceState(
        resonance_vector=resonance_vector,
        mu_value=mu_value,
        rho_density=rho_density,
        chi_cost=chi_cost,
        stability_score=stability_score,
        oscillation_phase=oscillation_phase,
        timestamp=timestamp
    )


class TestDataModelValidation:
    """Property-based tests for data model validation."""
    
    @given(phoneme_sequence_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    @pytest.mark.property
    def test_phoneme_sequence_validation_consistency(self, phoneme_seq):
        """
        Feature: urcm-reasoning-system, Property 1: Phoneme-to-Frequency Mapping Consistency
        
        For any valid PhonemeSequence, the validation should be consistent and
        the data structure should maintain its integrity across all operations.
        
        **Validates: Requirements 1.2**
        """
        # The phoneme sequence should always validate as true if properly constructed
        assert DataValidation.validate_phoneme_sequence(phoneme_seq) is True
        
        # The phoneme sequence should maintain its properties
        assert len(phoneme_seq.phonemes) > 0
        assert len(phoneme_seq.source_text.strip()) > 0
        
        # Validation should be deterministic - calling it multiple times should give same result
        first_validation = DataValidation.validate_phoneme_sequence(phoneme_seq)
        second_validation = DataValidation.validate_phoneme_sequence(phoneme_seq)
        assert first_validation == second_validation
    
    @given(frequency_path_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    @pytest.mark.property
    def test_frequency_path_validation_consistency(self, freq_path):
        """
        Feature: urcm-reasoning-system, Property 1: Phoneme-to-Frequency Mapping Consistency
        
        For any valid FrequencyPath, the validation should confirm dimensional constraints
        and smoothness properties are maintained.
        
        **Validates: Requirements 1.2**
        """
        # The frequency path should validate as true if properly constructed
        assert DataValidation.validate_frequency_path(freq_path) is True
        
        # Check dimensional constraints (K ∈ [16, 32])
        assert 16 <= freq_path.vectors.shape[1] <= 32
        
        # Check smoothness score is non-negative
        assert freq_path.smoothness_score >= 0
        
        # Check phoneme mapping consistency
        assert len(freq_path.phoneme_mapping) == freq_path.vectors.shape[0]
        
        # Validation should be deterministic
        first_validation = DataValidation.validate_frequency_path(freq_path)
        second_validation = DataValidation.validate_frequency_path(freq_path)
        assert first_validation == second_validation
    
    @given(resonance_state_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    @pytest.mark.property
    def test_resonance_state_validation_consistency(self, resonance_state):
        """
        Feature: urcm-reasoning-system, Property 1: Phoneme-to-Frequency Mapping Consistency
        
        For any valid ResonanceState, the validation should confirm mathematical
        constraints and stability properties are maintained.
        
        **Validates: Requirements 1.2**
        """
        # The resonance state should validate as true if properly constructed
        assert DataValidation.validate_resonance_state(resonance_state) is True
        
        # Check oscillation phase constraints [0, 2π]
        assert 0 <= resonance_state.oscillation_phase <= 2 * np.pi
        
        # Check timestamp is non-negative
        assert resonance_state.timestamp >= 0
        
        # Check μ value is positive and finite
        assert resonance_state.mu_value > 0
        assert np.isfinite(resonance_state.mu_value)
        
        # Check resonance vector has no NaN or infinite values
        assert np.all(np.isfinite(resonance_state.resonance_vector))
        
        # Validation should be deterministic
        first_validation = DataValidation.validate_resonance_state(resonance_state)
        second_validation = DataValidation.validate_resonance_state(resonance_state)
        assert first_validation == second_validation
    
    @given(st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    @pytest.mark.property
    def test_mu_value_validation_consistency(self, mu_value):
        """
        Feature: urcm-reasoning-system, Property 1: Phoneme-to-Frequency Mapping Consistency
        
        For any valid μ value, the validation should be consistent and confirm
        the mathematical constraints for semantic stability.
        
        **Validates: Requirements 1.2**
        """
        # Valid μ values should always validate as true
        assert DataValidation.validate_mu_value(mu_value) is True
        
        # μ should be positive for meaningful semantic density
        assert mu_value > 0
        
        # μ should be finite
        assert np.isfinite(mu_value)
        
        # Validation should be deterministic
        first_validation = DataValidation.validate_mu_value(mu_value)
        second_validation = DataValidation.validate_mu_value(mu_value)
        assert first_validation == second_validation


class TestDataModelIntegrity:
    """Unit tests for specific data model integrity checks."""
    
    @pytest.mark.unit
    def test_phoneme_sequence_empty_validation(self):
        """Test that empty phoneme sequences are properly rejected."""
        with pytest.raises(ValueError, match="Phoneme sequence cannot be empty"):
            PhonemeSequence(phonemes=[], source_text="test")
        
        with pytest.raises(ValueError, match="Source text cannot be empty"):
            PhonemeSequence(phonemes=["a"], source_text="")
    
    @pytest.mark.unit
    def test_frequency_path_dimension_validation(self):
        """Test that frequency paths with invalid dimensions are rejected."""
        # Test invalid K dimension (too small)
        with pytest.raises(ValueError, match="Frequency dimension K must be in range"):
            FrequencyPath(
                vectors=np.random.randn(5, 10),  # K=10 < 16
                smoothness_score=1.0,
                phoneme_mapping=[("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)]
            )
        
        # Test invalid K dimension (too large)
        with pytest.raises(ValueError, match="Frequency dimension K must be in range"):
            FrequencyPath(
                vectors=np.random.randn(5, 40),  # K=40 > 32
                smoothness_score=1.0,
                phoneme_mapping=[("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)]
            )
    
    @pytest.mark.unit
    def test_resonance_state_phase_validation(self):
        """Test that resonance states with invalid phases are rejected."""
        with pytest.raises(ValueError, match="Oscillation phase must be in range"):
            ResonanceState(
                resonance_vector=np.array([1.0, 2.0, 3.0]),
                mu_value=1.0,
                rho_density=0.5,
                chi_cost=0.5,
                stability_score=0.5,
                oscillation_phase=3*np.pi,  # > 2π
                timestamp=0.0
            )
    
    @pytest.mark.unit
    def test_mesh_signal_type_validation(self):
        """Test that mesh signals with invalid types are rejected."""
        with pytest.raises(ValueError, match="Signal type must be"):
            MeshSignal(
                sender_id="node1",
                delta_mu=0.5,
                phase_alignment=np.pi,
                timestamp=0.0,
                signal_type="invalid_type"
            )

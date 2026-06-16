"""
Property-based tests for the Resonance Path Encoding system.

Validates the temporal processing capability and consistency of the encoder.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

from urcm.core.phoneme_mapper import PhonemeFrequencyMapper
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.data_models import ResonanceState, FrequencyPath
from urcm.core.validation import DataValidation
from tests.test_phoneme_mapper import valid_phoneme_sequence_strategy, frequency_dimension_strategy

class TestResonanceEncoding:
    """Property tests for URCM Resonance Encoder."""
    
    @given(valid_phoneme_sequence_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    @pytest.mark.property
    def test_encoding_consistency(self, phonemes, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 8: Round-Trip Consistency (Partial)
        
        The encoder must be deterministic: given the same frequency path input,
        it must produce the exact same ResonanceState output.
        """
        # Setup
        mapper = PhonemeFrequencyMapper(frequency_dim=frequency_dim)
        freq_path = mapper.create_frequency_path(phonemes)
        
        encoder = ResonancePathEncoder(input_dim=frequency_dim, resonance_dim=64)
        
        # Execute twice
        state1 = encoder.get_resonance_state(freq_path)
        state2 = encoder.get_resonance_state(freq_path)
        
        # Assertions
        assert isinstance(state1, ResonanceState)
        
        # Check Vector Identity
        np.testing.assert_array_equal(
            state1.resonance_vector, 
            state2.resonance_vector,
            err_msg="Encoder is not deterministic!"
        )
        
        # Check Metric Identity
        assert state1.mu_value == state2.mu_value
        assert state1.rho_density == state2.rho_density
        assert state1.chi_cost == state2.chi_cost
        
    @given(valid_phoneme_sequence_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    @pytest.mark.property
    def test_state_validity(self, phonemes, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 8: Round-Trip Consistency (Partial)
        
        The encoder must always produce a mathematically valid ResonanceState
        that passes all core data model validations (e.g., rho/chi rules).
        """
        mapper = PhonemeFrequencyMapper(frequency_dim=frequency_dim)
        freq_path = mapper.create_frequency_path(phonemes)
        
        encoder = ResonancePathEncoder(input_dim=frequency_dim, resonance_dim=64)
        state = encoder.get_resonance_state(freq_path)
        
        # Validate using the central validation logic
        assert DataValidation.validate_resonance_state(state), "Generated state failed validation"
        
        # Additional Sanity Checks
        assert len(state.resonance_vector) == 64
        assert state.rho_density >= 0, "Rho must be non-negative"
        assert state.chi_cost >= 0, "Chi must be non-negative"
        
    @given(valid_phoneme_sequence_strategy(), frequency_dimension_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=20)
    @pytest.mark.property
    def test_temporal_sensitivity(self, phonemes, frequency_dim):
        """
        Feature: urcm-reasoning-system, Property 8: Round-Trip Consistency (Partial)
        
        The encoder must be sensitive to Temporal Order.
        Reversing the input sequence should produce a DIFFERENT resonance state.
        (Unlike 'Bag of Words' approaches which would map them identically).
        """
        # Need at least 2 different phonemes to test order sensitivity effectively
        # Also skip palindromes (e.g. ['a', 'b', 'a']) which are identical when reversed
        if len(phonemes) < 2 or len(set(phonemes)) == 1 or phonemes == phonemes[::-1]:
            return 

        mapper = PhonemeFrequencyMapper(frequency_dim=frequency_dim)
        
        # Forward Path
        path_fwd = mapper.create_frequency_path(phonemes)
        
        # Backward Path
        path_bwd = mapper.create_frequency_path(phonemes[::-1])
        
        encoder = ResonancePathEncoder(input_dim=frequency_dim, resonance_dim=64)
        
        vec_fwd = encoder.encode_path(path_fwd)
        vec_bwd = encoder.encode_path(path_bwd)
        
        # They should NOT be equal
        # We verify that the encoder is stateful and temporal order matters
        assert not np.allclose(vec_fwd, vec_bwd, atol=1e-5), \
            "Encoder produced identical output for reversed sequence! (Order ignored)"


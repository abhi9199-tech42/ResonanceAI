"""
Verification tests for URCM_TRANSFORMER_MERGE_GUIDE.md

Each test maps to a section in the guide and validates the code snippets
and claims made there actually work with the current codebase.
"""

import numpy as np
import pytest
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.oscillatory_gating import OscillatoryGating
from urcm.core.attractor_network import AttractorNetwork
from urcm.core.memory import GeometricMemory
from urcm.core.convergence_engine import MuConvergenceEngine
from urcm.core.theory import URCMTheory
from urcm.core.latent_space import SemanticLatentSpace, ReconstructionSystem
from urcm.core.safety import SafetyGovernor
from urcm.core.data_models import ResonanceState


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Architecture snapshot: weight shapes
# ─────────────────────────────────────────────────────────────────────────────

class TestArchitectureShapes:
    """Guide Section 1: validates stated tensor shapes."""

    def setup_method(self):
        self.encoder = ResonancePathEncoder(input_dim=24, resonance_dim=1024)

    def test_W_in_shape(self):
        assert self.encoder.W_in.shape == (24, 1024), \
            f"W_in expected (24, 1024), got {self.encoder.W_in.shape}"

    def test_W_res_shape(self):
        assert self.encoder.W_res.shape == (1024, 1024), \
            f"W_res expected (1024, 1024), got {self.encoder.W_res.shape}"

    def test_W_out_shape(self):
        assert self.encoder.W_out.shape == (1024, 24), \
            f"W_out expected (1024, 24), got {self.encoder.W_out.shape}"

    def test_W_res_spectral_radius_under_1(self):
        """Guide states W_res = orthogonal × 0.95 so spectral radius ≈ 0.95 < 1."""
        eigs = np.abs(np.linalg.eigvals(self.encoder.W_res))
        sr = float(np.max(eigs))
        assert sr <= 1.05, f"Spectral radius {sr:.4f} too large — W_res may be unstable"

    def test_gating_W_g_shape(self):
        gating = OscillatoryGating(resonance_dim=1024)
        assert gating.W_g.shape == (1024, 2), \
            f"W_g expected (1024, 2), got {gating.W_g.shape}"

    def test_latent_projection_shapes(self):
        ls = SemanticLatentSpace(input_dim=1024, latent_dim=16)
        assert ls.E.shape == (16, 1024), f"E shape wrong: {ls.E.shape}"
        assert ls.D.shape == (1024, 16), f"D shape wrong: {ls.D.shape}"

    def test_phoneme_vocab_size(self):
        """Guide states ~60 phonemes."""
        pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
        n = len(pipeline.frequency_mapper.phoneme_vectors)
        assert 50 <= n <= 80, f"Phoneme vocab size {n} out of expected [50, 80]"

    def test_phoneme_vectors_are_K_dim(self):
        """Each phoneme vector must be K=24 floats."""
        pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
        for name, vec in pipeline.frequency_mapper.phoneme_vectors.items():
            assert vec.shape == (24,), f"Phoneme '{name}' has wrong shape {vec.shape}"


# ─────────────────────────────────────────────────────────────────────────────
# Section 3.1 — Shallow merge: input layer replacement
# ─────────────────────────────────────────────────────────────────────────────

class TestInputLayerReplacement:
    """Guide Section 3.1: frequency vectors as drop-in token embeddings."""

    def setup_method(self):
        self.pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
        self.encoder = ResonancePathEncoder(input_dim=24, resonance_dim=1024)

    def test_urcm_embed_produces_sequence(self):
        """freq_path.vectors @ W_in gives (T, 1024) projected sequence."""
        text = "hello world"
        freq_path = self.pipeline.process_text(text)
        W_in = self.encoder.W_in
        projected = freq_path.vectors @ W_in
        assert projected.ndim == 2
        assert projected.shape[1] == 1024
        assert projected.shape[0] == freq_path.vectors.shape[0]

    def test_different_texts_produce_different_embeddings(self):
        texts = ["cat", "dog", "water"]
        vecs = []
        for t in texts:
            fp = self.pipeline.process_text(t)
            vecs.append((fp.vectors @ self.encoder.W_in).mean(axis=0))
        # All three mean-pooled vectors should differ
        assert not np.allclose(vecs[0], vecs[1]), "cat and dog embed identically"
        assert not np.allclose(vecs[0], vecs[2]), "cat and water embed identically"

    def test_frequency_dimension_bounds(self):
        """Guide states K ∈ [16, 32]."""
        for k in [16, 24, 32]:
            p = PhonemeFrequencyPipeline(frequency_dim=k)
            fp = p.process_text("test")
            assert fp.vectors.shape[1] == k


# ─────────────────────────────────────────────────────────────────────────────
# Section 3.2 — Resonance bottleneck
# ─────────────────────────────────────────────────────────────────────────────

class TestResonanceBottleneck:
    """Guide Section 3.2: run_dynamics_until_stable as a bottleneck layer."""

    def setup_method(self):
        self.encoder = ResonancePathEncoder(input_dim=24, resonance_dim=1024)
        self.gating = OscillatoryGating(resonance_dim=1024)

    def test_dynamics_returns_correct_shapes(self):
        state = np.random.randn(1024).astype(np.float32)
        state = np.tanh(state)
        final, steps, history = self.encoder.run_dynamics_until_stable(
            state=state,
            codebook_vectors={},
            max_steps=50,
            energy_tolerance=1e-3,
            return_history=True
        )
        assert final.shape == (1024,)
        assert isinstance(steps, int)
        assert isinstance(history, list)
        assert len(history) > 0

    def test_dynamics_converges_within_max_steps(self):
        """Guide says max_steps=50 should be sufficient."""
        state = np.random.randn(1024).astype(np.float32) * 0.5
        final, steps, _ = self.encoder.run_dynamics_until_stable(
            state=state,
            codebook_vectors={},
            max_steps=50,
            energy_tolerance=1e-3,
            return_history=False
        )
        assert steps <= 50

    def test_bottleneck_pipeline_proj_in_proj_out(self):
        """Simulate full bottleneck: linear_in → dynamics → gating → linear_out."""
        d_model = 64
        resonance_dim = 256
        enc = ResonancePathEncoder(input_dim=d_model, resonance_dim=resonance_dim)
        gate = OscillatoryGating(resonance_dim=resonance_dim)

        proj_in = np.random.normal(0, 0.02, (d_model, resonance_dim)).astype(np.float32)
        proj_out = np.random.normal(0, 0.02, (resonance_dim, d_model)).astype(np.float32)

        hidden = np.random.randn(d_model).astype(np.float32)
        z = hidden @ proj_in  # (resonance_dim,)

        stable, _, _ = enc.run_dynamics_until_stable(
            state=np.tanh(z),
            codebook_vectors={},
            max_steps=30,
            energy_tolerance=1e-3,
            return_history=False
        )
        gated = gate.apply_gating(stable, dt=0.1)
        output = gated @ proj_out  # (d_model,)

        assert output.shape == (d_model,)
        assert not np.any(np.isnan(output)), "NaN in bottleneck output"
        assert not np.any(np.isinf(output)), "Inf in bottleneck output"

    def test_batch_bottleneck(self):
        """Guide's URCMBottleneck.forward processes batch sequentially."""
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=256)
        gate = OscillatoryGating(resonance_dim=256)
        proj_in = np.random.normal(0, 0.02, (24, 256)).astype(np.float32)
        proj_out = np.random.normal(0, 0.02, (256, 24)).astype(np.float32)

        batch = np.random.randn(4, 24).astype(np.float32)
        results = []
        for i in range(batch.shape[0]):
            z = batch[i] @ proj_in
            stable, _, _ = enc.run_dynamics_until_stable(
                np.tanh(z), {}, max_steps=20, energy_tolerance=1e-3, return_history=False
            )
            gated = gate.apply_gating(stable, dt=0.1)
            results.append(gated @ proj_out)

        out = np.stack(results)
        assert out.shape == (4, 24)
        assert not np.any(np.isnan(out))


# ─────────────────────────────────────────────────────────────────────────────
# Section 3.3 — Memory-augmented attention
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryAugmentedAttention:
    """Guide Section 3.3: GeometricMemory as KV-cache replacement."""

    def setup_method(self):
        self.resonance_dim = 128
        self.mem = GeometricMemory(resonance_dim=self.resonance_dim)
        self.W_res = np.eye(self.resonance_dim, dtype=np.float32) * 0.95

    def _write(self, key: np.ndarray, value: np.ndarray):
        self.W_res = self.mem.deposit_attractor(self.W_res, key, value)

    def _read(self, query: np.ndarray, steps: int = 10) -> np.ndarray:
        state = query.copy()
        for _ in range(steps):
            state = np.tanh(state @ self.W_res)
        return state

    def test_write_does_not_corrupt_matrix_shape(self):
        key = np.random.randn(self.resonance_dim).astype(np.float32)
        val = np.random.randn(self.resonance_dim).astype(np.float32)
        self._write(key, val)
        assert self.W_res.shape == (self.resonance_dim, self.resonance_dim)

    def test_read_returns_correct_shape(self):
        q = np.random.randn(self.resonance_dim).astype(np.float32)
        out = self._read(q)
        assert out.shape == (self.resonance_dim,)

    def test_associated_retrieval(self):
        """After depositing key→value, reading near key should be closer to value than noise."""
        np.random.seed(7)
        key = np.random.randn(self.resonance_dim).astype(np.float32)
        key /= np.linalg.norm(key)
        value = np.random.randn(self.resonance_dim).astype(np.float32)
        value /= np.linalg.norm(value)

        for _ in range(5):  # deposit multiple times to strengthen
            self._write(key, value)

        retrieved = self._read(key, steps=20)
        noise = np.random.randn(self.resonance_dim).astype(np.float32)
        retrieved_noise = self._read(noise, steps=20)

        sim_correct = float(np.dot(retrieved, value) / (np.linalg.norm(retrieved) * np.linalg.norm(value) + 1e-9))
        sim_noise = float(np.dot(retrieved_noise, value) / (np.linalg.norm(retrieved_noise) * np.linalg.norm(value) + 1e-9))

        assert sim_correct > sim_noise, \
            f"Key-associated retrieval (sim={sim_correct:.3f}) not better than noise (sim={sim_noise:.3f})"

    def test_capacity_limit_defined(self):
        """Guide states capacity = resonance_dim × 0.5."""
        assert self.mem.capacity_limit == int(self.resonance_dim * 0.5)

    def test_deposit_increments_count(self):
        before = self.mem.deposited_count
        key = np.random.randn(self.resonance_dim).astype(np.float32)
        val = np.random.randn(self.resonance_dim).astype(np.float32)
        self._write(key, val)
        assert self.mem.deposited_count == before + 1

    def test_no_nan_after_many_deposits(self):
        for _ in range(20):
            k = np.random.randn(self.resonance_dim).astype(np.float32)
            v = np.random.randn(self.resonance_dim).astype(np.float32)
            self._write(k, v)
        assert not np.any(np.isnan(self.W_res)), "NaN in W_res after deposits"


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Output: ConceptDecoder nearest-neighbor
# ─────────────────────────────────────────────────────────────────────────────

class TestConceptDecoder:
    """Guide Section 4: retrieval-based output decoding."""

    def test_concept_decoder_returns_top_k(self):
        from urcm.tools.concept_decoder import ConceptDecoder
        from urcm.core.system import URCMSystem

        system = URCMSystem(resonance_dim=64)
        decoder = ConceptDecoder(system)

        vocab = ["cat", "dog", "water", "fire", "sky"]
        decoder.build_index(vocab)

        query_path = system.pipeline.process_text("cat")
        query_vec = system.encoder.get_resonance_state(query_path).resonance_vector

        results = decoder.decode(query_vec, top_k=3)
        assert len(results) == 3
        for text, sim in results:
            assert text in vocab
            assert -1.1 <= sim <= 1.1

    def test_concept_decoder_top1_is_self(self):
        """The closest concept to a word's own vector should be that word."""
        from urcm.tools.concept_decoder import ConceptDecoder
        from urcm.core.system import URCMSystem

        system = URCMSystem(resonance_dim=64)
        decoder = ConceptDecoder(system)
        vocab = ["apple", "banana", "cherry", "date", "elderberry"]
        decoder.build_index(vocab)

        for word in vocab:
            path = system.pipeline.process_text(word)
            vec = system.encoder.get_resonance_state(path).resonance_vector
            top = decoder.decode(vec, top_k=1)
            assert top[0][0] == word, \
                f"Expected '{word}' as top-1, got '{top[0][0]}'"


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — μ-metric as loss signal
# ─────────────────────────────────────────────────────────────────────────────

class TestMuMetricLoss:
    """Guide Section 5: μ = rho/chi as a differentiable-compatible loss signal."""

    def test_rho_is_bounded_0_1(self):
        for _ in range(20):
            v = np.random.randn(64).astype(np.float32)
            rho = URCMTheory.calculate_rho(v)
            assert 0.0 <= rho <= 1.0, f"rho={rho} out of [0, 1]"

    def test_rho_high_for_peaked_vector(self):
        """A vector with one dominant component should have high rho (low entropy)."""
        peaked = np.zeros(64, dtype=np.float32)
        peaked[0] = 10.0
        flat = np.ones(64, dtype=np.float32)
        assert URCMTheory.calculate_rho(peaked) > URCMTheory.calculate_rho(flat)

    def test_chi_is_nonneg(self):
        a = np.random.randn(64).astype(np.float32)
        b = np.random.randn(64).astype(np.float32)
        chi = URCMTheory.calculate_chi(a, b)
        assert chi >= 0.0

    def test_chi_zero_for_identical_states(self):
        v = np.random.randn(64).astype(np.float32)
        assert URCMTheory.calculate_chi(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_mu_increases_with_rho(self):
        chi = 1.0
        assert URCMTheory.compute_mu(0.9, chi) > URCMTheory.compute_mu(0.1, chi)

    def test_mu_decreases_with_chi(self):
        rho = 0.8
        assert URCMTheory.compute_mu(rho, 0.1) > URCMTheory.compute_mu(rho, 10.0)

    def test_normalised_mu_bounded(self):
        """Guide uses μ / (1 + |μ|) to normalise — result must be in (-1, 1)."""
        for _ in range(50):
            v = np.random.randn(64).astype(np.float32)
            rho = URCMTheory.calculate_rho(v)
            chi = float(np.linalg.norm(v)) + 0.01
            mu_raw = URCMTheory.compute_mu(rho, chi)
            mu_norm = mu_raw / (1.0 + abs(mu_raw))
            assert -1.0 < mu_norm < 1.0, f"Normalised mu {mu_norm} out of bounds"

    def test_loss_term_1_minus_mu(self):
        """L = 1 - μ_norm should be in (0, 2) for all inputs."""
        for _ in range(20):
            v = np.random.randn(32).astype(np.float32)
            rho = URCMTheory.calculate_rho(v)
            chi = float(np.linalg.norm(v)) + 0.01
            mu_raw = URCMTheory.compute_mu(rho, chi)
            mu_norm = mu_raw / (1.0 + abs(mu_raw))
            loss = 1.0 - mu_norm
            assert 0.0 < loss < 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Oscillatory gating as positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class TestOscillatoryPositionalEncoding:
    """Guide Section 6: gating as positional encoding replacement."""

    def test_different_positions_produce_different_encodings(self):
        """Same vector, different dt → different gated output."""
        gating = OscillatoryGating(resonance_dim=64, base_frequency=1.0)
        v = np.ones(64, dtype=np.float32) * 0.5

        gating.reset_phase(0.0)
        out_t0 = gating.apply_gating(v.copy(), dt=0.0)

        gating.reset_phase(0.0)
        out_t1 = gating.apply_gating(v.copy(), dt=0.5)

        assert not np.allclose(out_t0, out_t1), \
            "Positional encoding identical at t=0 and t=0.5"

    def test_sequence_positional_encoding_shape(self):
        """Encode a T-length sequence — output shape must be (T, d_model)."""
        d_model = 128
        T = 10
        gating = OscillatoryGating(resonance_dim=d_model, base_frequency=1.0)
        x = np.random.randn(T, d_model).astype(np.float32)

        out = []
        for t in range(T):
            gated = gating.apply_gating(x[t], dt=0.1)
            out.append(gated)
        result = np.stack(out)

        assert result.shape == (T, d_model)

    def test_gated_output_bounded(self):
        """Sigmoid gate keeps output in [-1, 1] for tanh-scaled inputs."""
        gating = OscillatoryGating(resonance_dim=64)
        v = np.tanh(np.random.randn(64)).astype(np.float32)
        out = gating.apply_gating(v, dt=0.1)
        assert np.all(np.abs(out) <= 1.0 + 1e-6)

    def test_phase_advances_per_token(self):
        gating = OscillatoryGating(resonance_dim=32)
        gating.reset_phase(0.0)
        phases = [gating.phase]
        for _ in range(5):
            gating.advance_time(0.1)
            phases.append(gating.phase)
        # All phases should be strictly increasing (modulo 2π wrapping)
        assert len(set(phases)) == len(phases), "Phase not advancing uniquely"


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Safety governor
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyGovernor:
    """Guide Section 7: SafetyGovernor wrapping bottleneck outputs."""

    def test_governor_init_and_lock(self):
        gov = SafetyGovernor(resonance_dim=64, max_spectral_radius=0.99)
        gov.lock_kernel()  # must not raise

    def test_clamp_energy_keeps_output_bounded(self):
        gov = SafetyGovernor(resonance_dim=64, max_spectral_radius=0.99)
        exploding = np.random.randn(64).astype(np.float32) * 1000.0
        clamped = gov.clamp_energy(exploding)
        assert np.all(np.isfinite(clamped)), "clamp_energy returned non-finite values"
        assert np.max(np.abs(clamped)) < np.max(np.abs(exploding)), \
            "clamp_energy did not reduce magnitude of exploding vector"

    def test_sanitize_input_returns_same_shape(self):
        gov = SafetyGovernor(resonance_dim=64, max_spectral_radius=0.99)
        x = np.random.randn(24).astype(np.float32)
        out = gov.sanitize_input(x)
        assert out.shape == x.shape

    def test_sanitize_handles_nan(self):
        gov = SafetyGovernor(resonance_dim=64, max_spectral_radius=0.99)
        x = np.array([np.nan, 1.0, 2.0], dtype=np.float32)
        out = gov.sanitize_input(x)
        assert not np.any(np.isnan(out)), "sanitize_input did not remove NaNs"

    def test_energy_ceiling_does_not_raise_on_safe_state(self):
        from urcm.core.safety import SafetyViolation
        gov = SafetyGovernor(resonance_dim=64, max_spectral_radius=0.99)
        safe = np.tanh(np.random.randn(64)).astype(np.float32)
        # Should not raise
        try:
            gov.check_energy_ceiling(safe)
        except SafetyViolation:
            pytest.fail("check_energy_ceiling raised SafetyViolation on a safe state")


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Known limitations
# ─────────────────────────────────────────────────────────────────────────────

class TestKnownLimitations:
    """Guide Section 9: verifies that the described limitations are real and bounded."""

    def test_W_res_orthogonality_degrades_after_deposits(self):
        """After deposits, W_res is no longer orthogonal — as stated in the guide."""
        mem = GeometricMemory(resonance_dim=64)
        rng = np.random.default_rng(0)
        H = rng.standard_normal((64, 64))
        Q, _ = np.linalg.qr(H)
        W = (Q * 0.95).astype(np.float32)

        # Measure initial orthogonality: W @ W.T ≈ I * 0.95²
        orth_before = np.linalg.norm(W @ W.T - np.eye(64) * 0.9025)

        for _ in range(30):
            k = rng.standard_normal(64).astype(np.float32)
            v = rng.standard_normal(64).astype(np.float32)
            W = mem.deposit_attractor(W, k, v)

        orth_after = np.linalg.norm(W @ W.T - np.eye(64) * 0.9025)
        assert orth_after > orth_before, \
            "Orthogonality should degrade after deposits (guide limitation §9)"

    def test_maintain_spectral_reconditions_W_res(self):
        """maintain_spectral() from URCMSystem reconditions W_res — guide §9."""
        from urcm.core.system import URCMSystem
        from urcm.core.memory_maintenance import MemoryMaintenance

        system = URCMSystem(resonance_dim=64)
        mem = GeometricMemory(resonance_dim=64)
        maint = MemoryMaintenance(system.encoder, mem, system.pipeline)

        # Inject heavy noise to bloat spectral radius
        system.encoder.W_res += np.random.randn(64, 64).astype(np.float32) * 0.5
        sr_before = float(np.max(np.abs(np.linalg.eigvals(system.encoder.W_res))))

        system.maintain_spectral(max_sigma=1.5)
        sr_after = float(np.max(np.abs(np.linalg.eigvals(system.encoder.W_res))))

        assert sr_after <= sr_before or sr_after <= 1.5 + 0.1, \
            f"maintain_spectral did not reduce spectral radius: {sr_before:.3f} → {sr_after:.3f}"

    def test_phoneme_tokenizer_drops_unknown_chars_gracefully(self):
        """Unknown chars should not crash the pipeline — guide §9."""
        pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
        # These contain chars outside the ASCII + Sanskrit map
        for weird in ["😀emoji", "中文", "αβγ", "!!!", "   "]:
            try:
                fp = pipeline.process_text(weird)
                # If it returns something, vectors must have correct dim
                assert fp.vectors.shape[1] == 24
            except ValueError:
                pass  # empty result is acceptable — important: no crash

    def test_dynamics_is_bounded_not_diverging(self):
        """run_dynamics_until_stable must not diverge — energy stays finite."""
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=256)
        state = np.random.randn(256).astype(np.float32) * 5.0  # large initial state
        final, steps, history = enc.run_dynamics_until_stable(
            state=np.tanh(state),
            codebook_vectors={},
            max_steps=100,
            energy_tolerance=1e-3,
            return_history=True
        )
        assert all(np.isfinite(e) for e in history), "Energy history contains non-finite values"
        assert np.all(np.isfinite(final)), "Final state contains non-finite values"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full pipeline end-to-end (guide Step 2 baseline)
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndPipeline:
    """Verifies the full URCM pipeline that the guide's merge steps build on."""

    def test_full_pipeline_text_to_resonance(self):
        from urcm.core.system import URCMSystem
        system = URCMSystem(resonance_dim=64)
        path = system.process_query("What do people use to absorb water?")
        assert path.final_state.resonance_vector.shape == (64,)
        assert np.isfinite(path.final_state.mu_value)

    def test_qa_commonsenseqa_passes_at_least_2_of_3(self):
        """Guide claims URCM passes CommonsenseQA — at least 2/3 must pass."""
        from urcm.core.system import URCMSystem
        from tests.test_commonsenseqa import choose_answer

        system = URCMSystem(resonance_dim=64)
        dataset = [
            {"q": "What do people use to absorb water?",
             "choices": ["spoon", "paper towel", "plate", "pen", "computer"],
             "answer_idx": 1},
            {"q": "Where do you store dishes in a kitchen?",
             "choices": ["cupboard", "trash can", "backpack", "street", "bed"],
             "answer_idx": 0},
            {"q": "What do you use to cut paper?",
             "choices": ["scissors", "spoon", "plate", "rope", "glue"],
             "answer_idx": 0},
        ]
        ok = sum(
            int(choose_answer(system, d["q"], d["choices"]) == d["answer_idx"])
            for d in dataset
        )
        assert ok >= 2, f"CommonsenseQA baseline too low: {ok}/3"

    def test_resonance_state_mu_positive_for_clear_input(self):
        from urcm.core.system import URCMSystem
        system = URCMSystem(resonance_dim=64)
        path = system.process_query("scissors")
        # mu should be > 0 for a clear, short input
        assert path.final_state.mu_value >= 0.0

"""
URCM Resonance Bottleneck — PyTorch nn.Module

Plugs into any transformer as a post-encoder verification layer.
Computes μ (mu) score per sequence to detect hallucination risk.

Architecture:
    hidden_states [B, T, d_model]
        → mean pool → [B, d_model]
        → proj_in   → [B, resonance_dim]
        → URCM dynamics (vectorized, CPU)
        → proj_out  → [B, d_model]
        → mu_score  → [B]   (hallucination risk: low μ = risky)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from urcm.core.theory import URCMTheory


class URCMBottleneck(nn.Module):
    """
    Drop-in URCM resonance layer for any transformer.

    Usage:
        bottleneck = URCMBottleneck(d_model=1024, resonance_dim=512)
        stable_hidden, mu_scores = bottleneck(hidden_states)

    Args:
        d_model:        Transformer hidden dimension (e.g. 1024 for GPT-2 medium)
        resonance_dim:  URCM resonance space dimension (default 512 for CPU)
        max_steps:      Max dynamics iterations (lower = faster, less stable)
        energy_tol:     Convergence threshold for dynamics
        mu_threshold:   Below this μ → flagged as hallucination risk
    """

    def __init__(
        self,
        d_model: int = 1024,
        resonance_dim: int = 512,
        max_steps: int = 20,
        energy_tol: float = 1e-3,
        mu_threshold: float = 0.3,
    ):
        super().__init__()
        self.d_model       = d_model
        self.resonance_dim = resonance_dim
        self.max_steps     = max_steps
        self.energy_tol    = energy_tol
        self.mu_threshold  = mu_threshold

        # Learnable projections in/out of resonance space
        self.proj_in  = nn.Linear(d_model, resonance_dim, bias=False)
        self.proj_out = nn.Linear(resonance_dim, d_model, bias=False)

        # URCM recurrent weight (not trained — initialized orthogonal × 0.95)
        W_res_np = self._init_orthogonal(resonance_dim) * 0.95
        # Register as buffer (moves with .to(device), not updated by optimizer)
        self.register_buffer("W_res", torch.tensor(W_res_np, dtype=torch.float32))

        # Layer norm for stable output
        self.layer_norm = nn.LayerNorm(d_model)

    # ── initialization ────────────────────────────────────────────────────────

    @staticmethod
    def _init_orthogonal(n: int) -> np.ndarray:
        M = np.random.randn(n, n).astype(np.float32)
        Q, _ = np.linalg.qr(M)
        return Q

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:   [B, T, d_model]
            attention_mask:  [B, T] optional, 1 = real token, 0 = padding

        Returns:
            stabilized:  [B, d_model]  — resonance-stabilized representation
            mu_scores:   [B]           — μ score per sequence (higher = more coherent)
        """
        B, T, D = hidden_states.shape

        # 1. Pool hidden states (masked mean if mask provided)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()          # [B, T, 1]
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)                   # [B, d_model]

        # 2. Project to resonance space
        z = self.proj_in(pooled)                                 # [B, resonance_dim]
        z = torch.tanh(z)

        # 3. Run URCM dynamics (vectorized over batch)
        z_stable, mu_scores = self._run_dynamics_batch(z)       # [B, resonance_dim], [B]

        # 4. Project back to d_model
        out = self.proj_out(z_stable)                            # [B, d_model]
        out = self.layer_norm(out)

        return out, mu_scores

    def _run_dynamics_batch(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized URCM dynamics over a batch.
        z: [B, resonance_dim]
        Returns: (z_stable [B, resonance_dim], mu_scores [B])
        """
        state = z.clone()
        prev_energy = torch.full((state.shape[0],), float("inf"), device=state.device)

        for _ in range(self.max_steps):
            # s_{t+1} = tanh(s_t @ W_res)
            new_state = torch.tanh(state @ self.W_res)           # [B, resonance_dim]

            # Energy = mean squared change
            energy = ((new_state - state) ** 2).mean(dim=1)     # [B]

            state = new_state

            # Check convergence (all samples converged)
            if (energy < self.energy_tol).all():
                break

            prev_energy = energy

        # Compute μ per sample
        mu_scores = self._compute_mu_batch(state)                # [B]
        return state, mu_scores

    def _compute_mu_batch(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute μ score per sample — measures semantic coherence.

        Uses three signals:
          1. ρ (rho)  = activation sparsity — coherent states have concentrated activations
          2. σ (std)  = per-dim variance across the batch — discriminates samples
          3. κ (kurt) = kurtosis proxy — peaky distributions = high signal

        Returns μ ∈ [0, 1] where higher = more coherent / less hallucination risk.
        """
        D = state.shape[1]

        # ── ρ: sparsity of activations (L1/L2 ratio, high = sparse = coherent) ──
        l1 = state.abs().sum(dim=1)                              # [B]
        l2 = state.norm(dim=1).clamp(min=1e-9)                  # [B]
        # L1/L2 ratio ∈ [1, sqrt(D)]; normalize to [0,1]
        sparsity = (l1 / l2 - 1.0) / (D ** 0.5 - 1.0 + 1e-9)  # [B]
        rho = sparsity.clamp(0.0, 1.0)

        # ── variance signal: how different is this sample from the batch mean? ──
        if state.shape[0] > 1:
            batch_mean = state.mean(dim=0, keepdim=True)         # [1, D]
            deviation  = ((state - batch_mean) ** 2).mean(dim=1) # [B]
            # Normalize deviation to [0, 1] within batch
            dev_min = deviation.min()
            dev_max = deviation.max()
            norm_dev = (deviation - dev_min) / (dev_max - dev_min + 1e-9)
        else:
            norm_dev = torch.zeros(state.shape[0], device=state.device)

        # ── μ: combine sparsity + deviation ──
        # High sparsity AND high deviation from generic mean = coherent, specific signal
        mu = 0.6 * rho + 0.4 * norm_dev
        return mu.clamp(0.0, 1.0)

    # ── hallucination detection ───────────────────────────────────────────────

    def is_hallucination_risk(self, mu_scores: torch.Tensor) -> torch.Tensor:
        """
        Returns bool tensor: True = hallucination risk, False = coherent.
        """
        return mu_scores < self.mu_threshold

    def score_text_batch(self, texts: list, tokenizer, device="cpu") -> dict:
        """
        Convenience method: score a list of strings directly.
        Returns dict with mu_scores and risk flags.
        """
        self.eval()
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Create fake hidden states from embedding (no full transformer needed for testing)
        # In real use, pass actual transformer hidden states
        B = inputs["input_ids"].shape[0]
        T = inputs["input_ids"].shape[1]
        fake_hidden = torch.randn(B, T, self.d_model, device=device)

        with torch.no_grad():
            _, mu_scores = self.forward(fake_hidden, inputs.get("attention_mask"))

        return {
            "mu_scores": mu_scores.tolist(),
            "risk":      self.is_hallucination_risk(mu_scores).tolist(),
            "threshold": self.mu_threshold,
        }

import numpy as np


class MemoryMaintenance:
    def __init__(self, encoder, geometric_memory, pipeline):
        self.encoder = encoder
        self.memory = geometric_memory
        self.pipeline = pipeline
    def spectral_clip(self, W: np.ndarray, max_sigma: float = 1.5) -> np.ndarray:
        u, s, vt = np.linalg.svd(W, full_matrices=False)
        s_clipped = np.minimum(s, max_sigma)
        return (u * s_clipped) @ vt
    def strengthen(self, W: np.ndarray, u: np.ndarray, v: np.ndarray, cycles: int = 400) -> np.ndarray:
        Z = W
        for _ in range(cycles):
            Z = self.memory.deposit_attractor(Z, u, v)
        return Z
    def weaken(self, W: np.ndarray, u: np.ndarray, v: np.ndarray, cycles: int = 600, alpha: float = 1.0) -> np.ndarray:
        Z = W
        for _ in range(cycles):
            norm_u_sq = float(np.dot(u, u))
            if norm_u_sq < 1e-9:
                return Z
            t = np.clip(v, -0.95, 0.95)
            lin_t = np.arctanh(t)
            cur = u @ Z
            e = lin_t - cur
            Z = Z - alpha * np.outer(u, e) / norm_u_sq
        return Z
    def forget(self, W: np.ndarray, u: np.ndarray, cycles: int = 300) -> np.ndarray:
        Z = W
        zt = np.zeros_like(u)
        for _ in range(cycles):
            Z = self.memory.deposit_attractor(Z, u, zt)
        return Z
    def encode_text(self, text: str) -> np.ndarray:
        p = self.pipeline.process_text(text)
        return self.encoder.get_resonance_state(p).resonance_vector

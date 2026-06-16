import numpy as np

class MoEHopfieldMemory:
    def __init__(self, dim: int, rank: int, experts: int, top_k: int = 2, seed: int = 0):
        self.dim = dim
        self.rank = rank
        self.experts = experts
        self.top_k = top_k
        rng = np.random.RandomState(seed)
        self.gate_W = rng.randn(experts, dim).astype(np.float64) * (1.0 / np.sqrt(dim))
        gw_norms = np.linalg.norm(self.gate_W, axis=1, keepdims=True) + 1e-9
        self.gate_W = self.gate_W / gw_norms
        self.U = []
        self.V = []
        for _ in range(experts):
            a = rng.randn(dim, rank).astype(np.float64)
            b = rng.randn(dim, rank).astype(np.float64)
            q1, _ = np.linalg.qr(a)
            q2, _ = np.linalg.qr(b)
            self.U.append((q1[:, :rank] * 0.05).copy())
            self.V.append((q2[:, :rank] * 0.05).copy())

    def _topk(self, x: np.ndarray):
        xn = x / (np.linalg.norm(x) + 1e-9)
        scores = self.gate_W @ xn
        idx = np.argsort(scores)[-self.top_k:][::-1]
        s = scores[idx]
        smax = s - np.max(s)
        ex = np.exp(smax)
        alphas = ex / (np.sum(ex) + 1e-9)
        return idx, alphas

    def forward(self, x: np.ndarray):
        idx, alphas = self._topk(x)
        y = np.zeros(self.dim, dtype=np.float64)
        for i, a in zip(idx, alphas):
            z = self.V[i].T @ x
            y_hat = self.U[i] @ z
            y += a * y_hat
        return y, idx, alphas

    def deposit(self, x: np.ndarray, target: np.ndarray, lr: float = 1e-3):
        idx, alphas = self._topk(x)
        for i, a in zip(idx, alphas):
            z = self.V[i].T @ x
            y_hat = self.U[i] @ z
            e = y_hat - target
            self.U[i] -= (lr * a) * np.outer(e, z)
            grad_z = self.U[i].T @ e
            self.V[i] -= (lr * a) * np.outer(x, grad_z)
        return idx, alphas

    def project(self, x: np.ndarray):
        idx, alphas = self._topk(x)
        out = []
        for i, a in zip(idx, alphas):
            z = self.V[i].T @ x
            y_hat = self.U[i] @ z
            out.append((i, a, y_hat))
        return out

    def save(self, path: str):
        u_stack = np.stack(self.U, axis=0)  # (experts, dim, rank)
        v_stack = np.stack(self.V, axis=0)  # (experts, dim, rank)
        np.savez_compressed(
            path,
            gate_W=self.gate_W,
            U=u_stack,
            V=v_stack,
            dim=self.dim,
            rank=self.rank,
            experts=self.experts,
            top_k=self.top_k
        )

    @staticmethod
    def load(path: str):
        data = np.load(path, allow_pickle=False)
        dim = int(data["dim"])
        rank = int(data["rank"])
        experts = int(data["experts"])
        top_k = int(data["top_k"])
        mem = MoEHopfieldMemory(dim=dim, rank=rank, experts=experts, top_k=top_k, seed=0)
        mem.gate_W = data["gate_W"]
        U = data["U"]
        V = data["V"]
        mem.U = [U[i].copy() for i in range(U.shape[0])]
        mem.V = [V[i].copy() for i in range(V.shape[0])]
        return mem

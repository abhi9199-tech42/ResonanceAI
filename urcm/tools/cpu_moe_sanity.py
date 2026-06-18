import numpy as np

from urcm.core.moe_hopfield import MoEHopfieldMemory


def cos(a, b):
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))

def main():
    dim = 256
    rank = 32
    experts = 8
    top_k = 2
    mem = MoEHopfieldMemory(dim=dim, rank=rank, experts=experts, top_k=top_k, seed=42)
    rng = np.random.RandomState(0)
    x = rng.randn(dim).astype(np.float64)
    x = x / (np.linalg.norm(x) + 1e-9)
    target = rng.randn(dim).astype(np.float64)
    target = target / (np.linalg.norm(target) + 1e-9)
    y0, idx0, a0 = mem.forward(x)
    base_sim = cos(y0, target)
    for _ in range(500):
        mem.deposit(x, target, lr=5e-3)
    y1, idx1, a1 = mem.forward(x)
    sim_after = cos(y1, target)
    out = {
        "selected_experts_before": [int(i) for i in idx0],
        "alphas_before": [float(a) for a in a0],
        "selected_experts_after": [int(i) for i in idx1],
        "alphas_after": [float(a) for a in a1],
        "cosine_before": base_sim,
        "cosine_after": sim_after
    }
    print(out)

if __name__ == "__main__":
    main()

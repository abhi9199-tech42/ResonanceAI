import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from urcm.core.moe_hopfield import MoEHopfieldMemory
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.resonance_encoder import ResonancePathEncoder


def cos(a, b):
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))

seed_pairs = [
    ("What do people use to absorb water?", "paper towel"),
    ("Where do you store dishes in a kitchen?", "cupboard"),
    ("What do you use to cut paper?", "scissors"),
    ("Where do you keep milk cold?", "refrigerator"),
]
qa_choices = {
    "What do people use to absorb water?": ["spoon","paper towel","plate","pen","computer"],
    "Where do you store dishes in a kitchen?": ["cupboard","trash can","backpack","street","bed"],
    "What do you use to cut paper?": ["scissors","spoon","plate","rope","glue"],
    "Where do you keep milk cold?": ["refrigerator","oven","desk","closet","backpack"],
}

def param_count(dim, rank, experts):
    return int(experts * 2 * dim * rank + experts * dim)

def load_dataset(path):
    pairs = []
    choices = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            q = o["q"]
            a = o["a"]
            pairs.append((q, a))
            if "choices" in o and isinstance(o["choices"], list) and len(o["choices"]) >= 2:
                choices[q] = o["choices"]
    return pairs, choices

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--dataset", type=str, default=None, help="JSONL with {q,a,choices?}")
    p.add_argument("--save", type=str, default="urcm_moe_cpu_ckpt.npz")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, default=None, help="Hint threads for BLAS")
    p.add_argument("--benchmark", action="store_true", help="Print timing/throughput metrics")
    p.add_argument("--target_updates", type=int, default=None, help="Total deposit calls to perform")
    p.add_argument("--interval_save", type=int, default=10000)
    p.add_argument("--show_progress", action="store_true")
    args = p.parse_args()

    if args.threads and args.threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
        try:
            from threadpoolctl import threadpool_limits  # type: ignore
            threadpool_limits(args.threads)
        except Exception:
            pass
    pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
    encoder = ResonancePathEncoder(input_dim=24, resonance_dim=args.dim)

    ckpt_path = Path(args.save)
    if args.resume and ckpt_path.exists():
        mem = MoEHopfieldMemory.load(str(ckpt_path))
        if mem.dim != args.dim or mem.rank != args.rank or mem.experts != args.experts:
            print("⚠️ Checkpoint shape mismatch; ignoring resume.", file=sys.stderr)
            mem = None
    else:
        mem = None
    if mem is None:
        try:
            mem = MoEHopfieldMemory(dim=args.dim, rank=args.rank, experts=args.experts, top_k=2, seed=42)
        except MemoryError:
            e = args.experts
            d = args.dim
            r = args.rank
            ok = False
            while not ok and (e >= 8 or d >= 128 or r >= 8):
                try:
                    mem = MoEHopfieldMemory(dim=d, rank=r, experts=e, top_k=2, seed=42)
                    ok = True
                except MemoryError:
                    if e > 8:
                        e = max(8, e // 2)
                    elif d > 128:
                        d = max(128, d // 2)
                    elif r > 8:
                        r = max(8, r // 2)
                    else:
                        break
            if not ok:
                raise
            print(json.dumps({"fallback": {"dim": d, "rank": r, "experts": e}}))
    stats = {"pairs": []}
    if args.dataset:
        pairs, extra_choices = load_dataset(args.dataset)
        if extra_choices:
            qa_choices.update(extra_choices)
    else:
        pairs = seed_pairs

    updates = 0
    t0 = time.time()
    if args.target_updates:
        idx = 0
        n = len(pairs)
        cos_before_map = {}
        for q, a in pairs[:min(8, len(pairs))]:
            fq = pipeline.process_text(q)
            u = encoder.encode_path(fq)
            y_pre, _, _ = mem.forward(u)
            v_pre = encoder.encode_path(pipeline.process_text(f"{q} {a}"))
            cos_before_map[q] = cos(y_pre, v_pre)
        while updates < args.target_updates:
            q, a = pairs[idx % n]
            fq = pipeline.process_text(q)
            fqa = pipeline.process_text(f"{q} {a}")
            u = encoder.encode_path(fq)
            v = encoder.encode_path(fqa)
            mem.deposit(u, v, lr=5e-3)
            updates += 1
            wrongs = [c for c in qa_choices.get(q, []) if c != a]
            for w in wrongs[:2]:
                if updates >= args.target_updates:
                    break
                fqw = pipeline.process_text(f"{q} {w}")
                uw = encoder.encode_path(fqw)
                mem.deposit(uw, -u, lr=3e-3)
                updates += 1
            if args.interval_save and updates % args.interval_save == 0:
                try:
                    mem.save(str(ckpt_path))
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning("Checkpoint save failed: %s", e)
                if args.benchmark:
                    elapsed_mid = max(1e-9, time.time() - t0)
                    print(json.dumps({"progress": {"updates": updates, "updates_per_sec": updates/elapsed_mid}}))
            idx += 1
        for q, a in pairs[:min(8, len(pairs))]:
            fq = pipeline.process_text(q)
            fqa = pipeline.process_text(f"{q} {a}")
            u = encoder.encode_path(fq)
            v = encoder.encode_path(fqa)
            y0, _, _ = mem.forward(u)
            cos_before = cos_before_map.get(q, cos(y0, v))
            stats["pairs"].append({"q": q, "a": a, "cos_before": cos_before, "cos_after": cos(y0, v)})
    else:
        for q, a in pairs:
            fq = pipeline.process_text(q)
            fqa = pipeline.process_text(f"{q} {a}")
            u = encoder.encode_path(fq)
            v = encoder.encode_path(fqa)
            y0, _, _ = mem.forward(u)
            base = cos(y0, v)
            for _ in range(args.epochs):
                mem.deposit(u, v, lr=5e-3)
                updates += 1
            y1, _, _ = mem.forward(u)
            after = cos(y1, v)
            wrongs = [c for c in qa_choices.get(q, []) if c != a]
            for w in wrongs[:2]:
                fqw = pipeline.process_text(f"{q} {w}")
                uw = encoder.encode_path(fqw)
                mem.deposit(uw, -u, lr=3e-3)
                updates += 1
            stats["pairs"].append({"q": q, "a": a, "cos_before": base, "cos_after": after})
            if args.show_progress:
                print(json.dumps(stats["pairs"][-1]))
    total_params = param_count(args.dim, args.rank, args.experts)
    t1 = time.time()
    elapsed = max(1e-9, t1 - t0)
    ups = updates / elapsed
    try:
        mem.save(str(ckpt_path))
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Final save failed: %s", e)
    out = {
        "params": total_params,
        "dim": args.dim,
        "rank": args.rank,
        "experts": args.experts,
        "pairs": stats["pairs"],
    }
    if args.benchmark:
        out["timing"] = {"elapsed_sec": elapsed, "updates": updates, "updates_per_sec": ups}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

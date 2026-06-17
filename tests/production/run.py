"""
Phase 2 — Fair Benchmarking: η-convergence hallucination detection evaluation.
Trains URCM on held-out train KB, evaluates on length-controlled test data,
reports AUROC, ECE, and calibration metrics.

Usage:
    python tests/production/run.py                          # run benchmark
    python tests/production/run.py --report                 # run + print report
    python tests/production/run.py --save results.json      # save to file
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from urcm.core.system import URCMSystem
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.theory import URCMTheory

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_json(name):
    with open(os.path.join(ROOT, name), "r", encoding="utf-8") as f:
        return json.load(f)


def train_from_kb(kb_pairs):
    """Create a fresh URCMSystem and populate its hippocampus from KB pairs."""
    system = URCMSystem(resonance_dim=2048, use_wave_dynamics=False)
    pipeline = PhonemeFrequencyPipeline(frequency_dim=24)

    for question, answer, wrongs in kb_pairs:
        system.learn_concept_oneshot(question, answer)

    return system


def encode_text(system, text):
    fp = system.pipeline.process_text(text)
    state = system.encoder.get_resonance_state(fp)
    return state.resonance_vector


def run_single_test(system, question, variant_text):
    """Run hallucination detection on one variant."""
    return system.detect_hallucination(variant_text, top_k=5)


def compute_auroc(scores, labels):
    """Compute AUROC given scores (higher = more factual) and binary labels (1 = factual)."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def compute_ece(scores, labels, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = np.where((scores >= lo) & (scores < hi))[0]
        if len(in_bin) == 0:
            continue
        bin_conf = np.mean(scores[in_bin])
        bin_acc = np.mean(labels[in_bin])
        ece += (len(in_bin) / len(scores)) * abs(bin_conf - bin_acc)
    return float(ece)


def build_reliability(scores, labels, n_bins=10):
    """Build reliability diagram data."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = np.where((scores >= lo) & (scores < hi))[0]
        if len(in_bin) == 0:
            continue
        bins.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "count": int(len(in_bin)),
            "avg_confidence": float(np.mean(scores[in_bin])),
            "accuracy": float(np.mean(labels[in_bin])),
        })
    return bins


def evaluate(system, test_data, silent=False):
    """Run evaluation on test_data and return metrics."""
    results = []
    for test in test_data["tests"]:
        question = test["question"]

        for variant_key in ["short_factual", "long_factual", "short_hallucination", "long_hallucination"]:
            text = test[variant_key]
            label = 1 if "factual" in variant_key else 0
            length = "short" if "short" in variant_key else "long"

            t0 = time.time()
            det = run_single_test(system, question, text)
            elapsed = time.time() - t0

            results.append({
                "question": question,
                "variant": variant_key,
                "text": text,
                "label": label,
                "length": length,
                "confidence": det["confidence"],
                "mu_value": det["mu_value"],
                "rho": det["rho"],
                "chi": det["chi"],
                "nn_label": det["nn_label"],
                "paradox": det.get("paradox_detected", False),
                "latency_ms": round(elapsed * 1000, 1),
            })

            if not silent:
                tag = "FACT" if label == 1 else "HALL"
                print(f"  [{tag}] {variant_key:25s} conf={det['confidence']:.3f} mu={det['mu_value']:.2e} rho={det['rho']:.3f} chi={det['chi']:.3f} nn={det['nn_label']}")

    # --- Compute metrics ---
    scores = np.array([r["confidence"] for r in results])
    labels = np.array([r["label"] for r in results])

    all_auroc = compute_auroc(scores, labels)
    all_ece = compute_ece(scores, labels)

    # Per-length bucket
    short_mask = np.array([r["length"] == "short" for r in results])
    long_mask = np.array([r["length"] == "long" for r in results])

    short_auroc = compute_auroc(scores[short_mask], labels[short_mask]) if np.sum(short_mask) > 1 else 0.0
    long_auroc = compute_auroc(scores[long_mask], labels[long_mask]) if np.sum(long_mask) > 1 else 0.0

    reliability = build_reliability(scores, labels)

    metrics = {
        "total_tests": len(results),
        "auroc_all": all_auroc,
        "auroc_short": short_auroc,
        "auroc_long": long_auroc,
        "ece_all": all_ece,
        "calibration": reliability,
        "mean_confidence_factual": float(np.mean(scores[labels == 1])),
        "mean_confidence_hallucination": float(np.mean(scores[labels == 0])),
        "mean_mu_factual": float(np.mean([r["mu_value"] for r in results if r["label"] == 1])),
        "mean_mu_hallucination": float(np.mean([r["mu_value"] for r in results if r["label"] == 0])),
        "results": results,
    }

    return metrics


def print_report(metrics):
    """Print a human-readable report."""
    print("\n" + "=" * 60)
    print("  URCM HALLUCINATION DETECTION BENCHMARK REPORT")
    print("=" * 60)
    print(f"  Total test samples:  {metrics['total_tests']}")
    print(f"  AUROC (all):         {metrics['auroc_all']:.4f}")
    print(f"  AUROC (short only):  {metrics['auroc_short']:.4f}")
    print(f"  AUROC (long only):   {metrics['auroc_long']:.4f}")
    print(f"  ECE:                 {metrics['ece_all']:.4f}")
    print(f"  Mean conf factual:   {metrics['mean_confidence_factual']:.4f}")
    print(f"  Mean conf hallucin:  {metrics['mean_confidence_hallucination']:.4f}")
    print(f"  Mean mu factual:     {metrics['mean_mu_factual']:.2e}")
    print(f"  Mean mu hallucin:    {metrics['mean_mu_hallucination']:.2e}")
    print(f"  Latency/query:       {np.mean([r['latency_ms'] for r in metrics['results']]):.1f}ms")
    print("-" * 60)

    print("\n  Calibration:")
    for b in metrics["calibration"]:
        print(f"    bin [{b['bin_lo']:.1f}, {b['bin_hi']:.1f}): n={b['count']:3d} conf={b['avg_confidence']:.3f} acc={b['accuracy']:.3f} gap={abs(b['avg_confidence'] - b['accuracy']):.3f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Fair Benchmarking for URCM")
    parser.add_argument("--report", action="store_true", help="Print report to stdout")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--silent", action="store_true", help="Suppress per-sample output")
    args = parser.parse_args()

    print("Loading test data and KB...")
    kb_train = load_json("kb_train.json")
    test_data = load_json("test_data.json")

    print(f"Train KB: {len(kb_train['pairs'])} pairs")
    print(f"Test data: {len(test_data['tests'])} questions, {sum(1 for t in test_data['tests'] for k in ['short_factual','long_factual','short_hallucination','long_hallucination'])} variants")

    print("\nTraining URCM from KB (this will overwrite urcm_weights.pkl)...")
    # We train by invoking train_2048.py with modified QA list
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_2048", os.path.join(ROOT, "..", "..", "train_2048.py"))
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    # Override COMMONSENSE_QA and re-run main
    train_mod.COMMONSENSE_QA = kb_train["pairs"]
    train_mod.main()

    print("\nPerforming evaluation...")
    # Reload system with new weights
    system = URCMSystem(resonance_dim=2048, use_wave_dynamics=False)
    metrics = evaluate(system, test_data, silent=args.silent)

    if args.report or not args.silent:
        print_report(metrics)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.save}")

    return metrics


if __name__ == "__main__":
    main()

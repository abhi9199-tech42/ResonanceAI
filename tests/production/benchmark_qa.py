"""Full benchmark: verify_qa on all 17 test questions (68 variants)."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from urcm.core.system import URCMSystem
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT, "test_data.json")) as f:
    test_data = json.load(f)

system = URCMSystem(resonance_dim=2048, use_wave_dynamics=False)

results = []
for test in test_data["tests"]:
    q = test["question"]
    for key in ["short_factual", "long_factual", "short_hallucination", "long_hallucination"]:
        text = test[key]
        label = 1 if "factual" in key else 0
        length = "short" if "short" in key else "long"
        det = system.verify_qa(q, text)
        results.append({
            **det,
            "label": label,
            "length": length,
            "variant": key,
            "question": q,
        })

scores = np.array([r["confidence"] for r in results])
labels = np.array([r["label"] for r in results])
short = np.array([r["length"] == "short" for r in results])
long_ = np.array([r["length"] == "long" for r in results])

auroc_all = roc_auc_score(labels, scores)
auroc_short = roc_auc_score(labels[short], scores[short]) if sum(short) > 1 else 0.0
auroc_long = roc_auc_score(labels[long_], scores[long_]) if sum(long_) > 1 else 0.0

print("=" * 60)
print("  VERIFY QA BENCHMARK — Full 68-sample test")
print("=" * 60)
print("  Samples:       %d (%d short, %d long)" % (len(results), sum(short), sum(long_)))
print("  AUROC (all):   %.4f" % auroc_all)
print("  AUROC (short): %.4f" % auroc_short)
print("  AUROC (long):  %.4f" % auroc_long)
print("  Mean factual:  %.4f" % scores[labels==1].mean())
print("  Mean halluc:   %.4f" % scores[labels==0].mean())
print("  Correct dir:   %d/68" % sum(
    (r["confidence"] > 0.5) == (r["label"] == 1) for r in results
))
print()

print("%-45s %7s %7s %7s %7s" % ("Question","ShortF","ShortH","LongF","LongH"))
for test in test_data["tests"]:
    q = test["question"][:42]
    sf = [r["confidence"] for r in results if r["question"]==test["question"] and r["variant"]=="short_factual"]
    sh = [r["confidence"] for r in results if r["question"]==test["question"] and r["variant"]=="short_hallucination"]
    lf = [r["confidence"] for r in results if r["question"]==test["question"] and r["variant"]=="long_factual"]
    lh = [r["confidence"] for r in results if r["question"]==test["question"] and r["variant"]=="long_hallucination"]
    ok_short = sf[0] > sh[0] if sf and sh else False
    ok_long = lf[0] > lh[0] if lf and lh else False
    print("%-45s %7.3f %7.3f %7.3f %7.3f  %s%s" % (
        q, sf[0] if sf else 0, sh[0] if sh else 0,
        lf[0] if lf else 0, lh[0] if lh else 0,
        "Y" if ok_short else "N", "Y" if ok_long else "N"))

# Save
out = {
    "auroc_all": auroc_all,
    "auroc_short": auroc_short,
    "auroc_long": auroc_long,
    "mean_conf_factual": float(scores[labels==1].mean()),
    "mean_conf_hallucination": float(scores[labels==0].mean()),
    "correct_count": int(sum((r["confidence"] > 0.5) == (r["label"] == 1) for r in results)),
    "total": len(results),
    "method": "verify_qa",
}
with open(os.path.join(ROOT, "benchmark_verify_qa.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to benchmark_verify_qa.json")
print(json.dumps(out, indent=2))

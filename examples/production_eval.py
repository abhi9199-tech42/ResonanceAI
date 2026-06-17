"""
Production evaluation: test URCM against real LLM outputs.

Requires: pip install openai (for ChatGPT/Claude) OR transformers (for local LLMs).

Usage:
  python -m examples.production_eval
"""
import sys, os, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from urcm.core.system import URCMSystem

# 20 test questions with known correct answers (held-out from URCM training)
TEST_SET = [
    ("What absorbs water?", "paper towel"),
    ("What cuts paper?", "scissors"),
    ("What tells time?", "clock"),
    ("What do you sleep on?", "bed"),
    ("What boils water?", "kettle"),
    ("What cleans teeth?", "toothbrush"),
    ("What do you write with?", "pen"),
    ("Where to store dishes?", "cupboard"),
    ("What do you eat soup with?", "spoon"),
    ("What keeps milk cold?", "refrigerator"),
    ("What lights a dark room?", "lamp"),
    ("What do you carry books in?", "backpack"),
    ("What measures temperature?", "thermometer"),
    ("What do you use to call someone?", "phone"),
    ("What do you sit on?", "chair"),
    ("What holds water for drinking?", "cup"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What gas do plants absorb?", "carbon dioxide"),
    ("What force keeps us on the ground?", "gravity"),
]


def generate_hallucinations_gpt2(questions):
    """Generate hallucinated answers using GPT-2."""
    from transformers import pipeline
    gpt2 = pipeline("text-generation", model="distilgpt2", device_map=None)
    hallucinations = []
    for q, correct in questions:
        for _ in range(5):
            out = gpt2(f"Q: {q}\nA:", max_new_tokens=20, do_sample=True, temperature=0.9)
            ans = out[0]['generated_text'].split("A:")[-1].strip()[:80]
            if ans and correct.lower() not in ans.lower() and len(ans) > 5:
                hallucinations.append(ans)
                break
        else:
            hallucinations.append("I am not sure about this")
    return hallucinations


def evaluate_urcm(system, samples, name="URCM"):
    """Evaluate URCM on test samples and return metrics."""
    scores = []
    for text, label in samples:
        r = system.detect_hallucination(text)
        scores.append((r["confidence"], label))

    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    labels = np.array([s[1] for s in scores])
    preds = np.array([s[0] for s in scores])
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    precs, recs, ths = precision_recall_curve(labels, preds)
    f1s = 2 * precs * recs / (precs + recs + 1e-9)
    best_f1 = f1s.max()
    fact_mean = preds[labels == 1].mean()
    hall_mean = preds[labels == 0].mean()

    print(f"\n{name}:")
    print(f"  AUROC:          {auc:.3f}")
    print(f"  Avg Precision:  {ap:.3f}")
    print(f"  Best F1:        {best_f1:.3f}")
    print(f"  factual μ:      {fact_mean:.3f}")
    print(f"  hallucinated μ: {hall_mean:.3f}")
    print(f"  gap:            {fact_mean - hall_mean:+.3f}")

    return {"auc": auc, "ap": ap, "f1": best_f1}


def main():
    print("=" * 55)
    print("Production Hallucination Detection Evaluation")
    print("=" * 55)
    print(f"Test set: {len(TEST_SET)} questions")

    # Generate hallucinations
    print("\nGenerating GPT-2 hallucinations...")
    hallucinations = generate_hallucinations_gpt2(TEST_SET)
    print(f"Generated {len(hallucinations)} hallucinated answers")

    # Build test samples
    samples = []
    for (q, correct), hall in zip(TEST_SET, hallucinations):
        samples.append((correct, 1))    # factual
        samples.append((hall, 0))       # hallucinated

    print(f"Total samples: {len(samples)} ({len(samples)//2} factual, {len(samples)//2} hallucinated)")

    # Evaluate URCM 62-pair
    print("\nLoading URCM 62-pair...")
    system_62 = URCMSystem(resonance_dim=2048)
    metrics_62 = evaluate_urcm(system_62, samples, "URCM 62-pair")

    # Evaluate URCM BERT
    try:
        print("\nLoading URCM BERT...")
        system_bert = URCMSystem(resonance_dim=2048, load_pretrained="bert-base-uncased")
        metrics_bert = evaluate_urcm(system_bert, samples, "URCM BERT")
    except Exception as e:
        print(f"BERT weights not available: {e}")
        metrics_bert = None

    # Evaluate S-BERT baseline (if available)
    try:
        from sentence_transformers import SentenceTransformer, util
        kb_answers = [a for q, a in TEST_SET]
        print("\nLoading Sentence-BERT...")
        sem = SentenceTransformer("all-MiniLM-L6-v2")
        kb_embs = sem.encode(kb_answers, convert_to_tensor=True)
        bert_scores = []
        for text, label in samples:
            emb = sem.encode(text, convert_to_tensor=True)
            sims = util.cos_sim(emb, kb_embs)[0].cpu().numpy()
            bert_scores.append((float(sims.max()), label))
        metrics_bert_sem = evaluate_urcm(None, bert_scores, "S-BERT")
    except Exception as e:
        print(f"S-BERT not available: {e}")
        metrics_bert_sem = None

    # Summary
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"{'Method':25s} {'AUROC':>8s} {'AP':>8s} {'F1':>8s}")
    print("-" * 50)
    print(f"{'URCM 62-pair':25s} {metrics_62['auc']:>8.3f} {metrics_62['ap']:>8.3f} {metrics_62['f1']:>8.3f}")
    if metrics_bert:
        print(f"{'URCM BERT':25s} {metrics_bert['auc']:>8.3f} {metrics_bert['ap']:>8.3f} {metrics_bert['f1']:>8.3f}")
    if metrics_bert_sem:
        print(f"{'S-BERT':25s} {metrics_bert_sem['auc']:>8.3f} {metrics_bert_sem['ap']:>8.3f} {metrics_bert_sem['f1']:>8.3f}")

    # Save results
    results = {
        "n_test": len(TEST_SET),
        "n_samples": len(samples),
        "62_pair": metrics_62,
        "bert": metrics_bert,
        "sbert": metrics_bert_sem,
    }
    with open("production_eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to production_eval_results.json")


if __name__ == "__main__":
    main()

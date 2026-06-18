"""
ResonanceAI CLI — hallucination detection and QA via phoneme resonance.
"""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_detect(args):
    """Detect hallucination in a given text (the core use case)."""
    kwargs = {"resonance_dim": args.dim}
    if args.bert:
        kwargs["load_pretrained"] = "bert-base-uncased"
    from urcm.core.system import URCMSystem
    system = URCMSystem(**kwargs)
    result = system.detect_hallucination(args.text, top_k=args.top_k)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        label = "LIKELY CORRECT" if result["confidence"] > args.threshold else "LIKELY HALLUCINATION"
        print(f"Confidence:     {result['confidence']:.3f}")
        print(f"Raw similarity: {result.get('raw_cosine', 0):.3f}")
        print(f"Nearest match:  {result['nn_label']}")
        print(f"Verdict:        {label}")


def cmd_qa(args):
    """Answer a question using the URCM hippocampus."""
    from urcm.core.system import URCMSystem
    system = URCMSystem(resonance_dim=args.dim)
    choices = [c.strip() for c in args.choices.split(",")] if args.choices else []
    result = system.solve_qa_right_brain(args.question, choices)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Question: {args.question}")
        if "details" in result:
            for d in result["details"]:
                print(f"  {d['choice']:20s} score={d['score']:.3f}")
        print(f"Answer:   {result.get('winner', 'N/A')}")


def cmd_benchmark(args):
    """Run hallucination detection benchmark vs S-BERT."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Import and run the comprehensive benchmark
    from urcm.core.system import URCMSystem
    try:
        from sentence_transformers import SentenceTransformer, util
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError as e:
        print(f"Missing optional dependency: {e}")
        print("Install with: pip install resonanceai[benchmark]")
        sys.exit(1)

    # Small test set for quick validation
    test_pairs = [
        ("What absorbs water?", "paper towel"),
        ("What cuts paper?", "scissors"),
        ("What tells time?", "clock"),
        ("What do you sleep on?", "bed"),
        ("What boils water?", "kettle"),
    ]
    hallucinated = [
        "The absorbent material is made of fibers",
        "You can never know what cuts it",
        "It depends on how long the number is",
        "There is a lot of sleep to be had",
        "The water gets hot eventually",
    ]

    samples = []
    for (q, a), h in zip(test_pairs, hallucinated):
        samples.append((a, 1, q))
        samples.append((h, 0, q))

    # URCM
    system = URCMSystem(resonance_dim=args.dim)
    urcm_scores = []
    for text, label, q in samples:
        r = system.detect_hallucination(text)
        urcm_scores.append((r["confidence"], label))

    # S-BERT
    sem = SentenceTransformer("all-MiniLM-L6-v2")
    kb_answers = [a for q, a in test_pairs]
    kb_embs = sem.encode(kb_answers, convert_to_tensor=True)
    bert_scores = []
    for text, label, q in samples:
        emb = sem.encode(text, convert_to_tensor=True)
        sims = util.cos_sim(emb, kb_embs)[0].cpu().numpy()
        bert_scores.append((float(sims.max()), label))

    def print_metrics(name, scores):
        labels = np.array([s[1] for s in scores])
        preds = np.array([s[0] for s in scores])
        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        print(f"  {name:20s} AUROC={auc:.3f}  AP={ap:.3f}")

    print("Benchmark (5 pairs, 10 samples):")
    print_metrics("URCM", urcm_scores)
    print_metrics("S-BERT", bert_scores)


def main():
    parser = argparse.ArgumentParser(
        prog="resonanceai",
        description="Hallucination detection via phoneme-resonance dynamics"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    sub = parser.add_subparsers(dest="cmd")

    # detect
    p = sub.add_parser("detect", help="Detect hallucination in text")
    p.add_argument("text", help="Text to check")
    p.add_argument("--dim", type=int, default=2048, help="Resonance dimension")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.65, help="Decision threshold")
    p.add_argument("--bert", action="store_true", help="Use BERT-converted weights")
    p.set_defaults(func=cmd_detect)

    # qa
    p = sub.add_parser("qa", help="Answer a question via hippocampus")
    p.add_argument("question", help="Question to answer")
    p.add_argument("--choices", help="Comma-separated candidate answers")
    p.add_argument("--dim", type=int, default=2048)
    p.set_defaults(func=cmd_qa)

    # benchmark
    p = sub.add_parser("benchmark", help="Run quick benchmark vs S-BERT")
    p.add_argument("--dim", type=int, default=2048)
    p.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()

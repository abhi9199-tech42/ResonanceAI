import re
from pathlib import Path


def test_hi_document_quality_structure_and_readability():
    doc_path = Path("urcm/12345test/hi")
    assert doc_path.exists()
    text = doc_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    assert len(lines) >= 150

    tiers = [
        "TIER 1: BASIC COGNITION",
        "TIER 2: REASONING",
        "TIER 3: LEARNING",
        "TIER 4: CREATIVITY",
        "TIER 5: SOCIAL INTELLIGENCE",
        "TIER 6: AUTONOMY & AGENCY",
        "TIER 7: GENERALIZATION",
        "TIER 8: SAFETY & ALIGNMENT",
    ]
    for t in tiers:
        assert t in text

    indices = [
        "1.1",
        "1.2",
        "1.3",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "3.1",
        "3.2",
        "3.3",
        "3.4",
        "4.1",
        "4.2",
        "4.3",
        "5.1",
        "5.2",
        "5.3",
        "6.1",
        "6.2",
        "6.3",
        "6.4",
        "7.1",
        "7.2",
        "7.3",
        "8.1",
        "8.2",
        "8.3",
    ]
    for idx in indices:
        assert re.search(rf"{re.escape(idx)}\s", text)

    samples = [
        "Image recognition",
        "Audio processing",
        "Video understanding",
        "Multi-modal fusion",
        "Arithmetic:",
        "Algebra:",
        "Geometry:",
        "Calculus:",
        "Statistics:",
        "Physical intuition:",
        "Causal reasoning:",
        "Temporal reasoning:",
        "Spatial reasoning:",
        "Social norms:",
        "Counterfactual thinking:",
        "Mental simulation:",
        "Joke understanding:",
        "Theory of Mind",
        "Communication",
        "Collaboration",
        "Goal Formation",
        "Planning",
        "Execution",
        "Self-Awareness",
        "Cross-Task Performance",
        "Domain Independence",
        "Real-World Interaction",
        "Value Alignment",
        "Robustness",
        "Controllability",
        "Turing Test",
        "Winograd Schema Challenge",
        "Coffee Test",
        "College Student Test",
        "Employment Test",
        "Science Test",
    ]
    for s in samples:
        assert s in text

    max_len = max(len(line) for line in lines if line.strip())
    assert max_len <= 200

    words = re.findall(r"\w+", text)
    sentences = [s for s in re.split(r"[.!?\n]+", text) if s.strip()]
    assert len(words) > 100
    assert len(sentences) > 20
    avg_words_per_sentence = len(words) / len(sentences)
    assert 5 <= avg_words_per_sentence <= 40

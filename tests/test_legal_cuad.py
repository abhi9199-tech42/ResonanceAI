import numpy as np
import pytest
from urcm.core.system import URCMSystem

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))

def evaluate_legal_query(system: URCMSystem, context: str, question: str, choices: list[str]) -> int:
    """
    Evaluates a legal query using Right Brain resonance and Context Anchoring.
    """
    # Use context + question as the thought prompt
    full_prompt = f"{context} {question}"
    
    # Extract keywords from BOTH context and question
    anchors = system.get_context_anchors(full_prompt, top_k=5)
    kw = [a[1] for a in anchors]
    
    result = system.solve_qa_right_brain(full_prompt, choices, context_keywords=kw)
    winner = result["winner"]
    
    return choices.index(winner)

@pytest.mark.legal
def test_cuad_legal_reasoning():
    system = URCMSystem(resonance_dim=1024, max_steps=10)
    
    # CUAD-inspired Legal/Contractual Queries
    dataset = [
        {
            "context": "The parties shall maintain the confidentiality of all Proprietary Information and shall not disclose it to any third party without prior written consent.",
            "q": "What is the primary obligation of this clause?",
            "choices": ["Confidentiality", "Payment terms", "Termination", "Indemnification", "Governing Law"],
            "answer_idx": 0
        },
        {
            "context": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.",
            "q": "What does this clause specify?",
            "choices": ["Governing Law", "Force Majeure", "Non-compete", "Arbitration", "Change of Control"],
            "answer_idx": 0
        },
        {
            "context": "Either party may terminate this Agreement upon thirty (30) days' prior written notice to the other party.",
            "q": "What is the notice period for termination?",
            "choices": ["30 days", "15 days", "60 days", "90 days", "Immediate"],
            "answer_idx": 0
        },
        {
            "context": "The Consultant shall be an independent contractor and not an employee, agent, or partner of the Company.",
            "q": "How is the Consultant classified?",
            "choices": ["Independent Contractor", "Full-time Employee", "Legal Partner", "General Agent", "Joint Venturer"],
            "answer_idx": 0
        },
        {
            "context": "Each party shall indemnify, defend, and hold harmless the other party from any claims arising out of its negligence.",
            "q": "Which legal concept is described here?",
            "choices": ["Indemnification", "Limitation of Liability", "Warranty", "Severability", "Assignment"],
            "answer_idx": 0
        },
        {
            "context": "If a court finds any provision of this Agreement invalid, the remaining provisions shall remain in full force.",
            "q": "What type of clause is this?",
            "choices": ["Severability", "Entire Agreement", "Survival", "Counterparts", "Notice"],
            "answer_idx": 0
        },
        {
            "context": "No party may assign its rights or delegate its duties under this Agreement without the other party's consent.",
            "q": "What does this clause restrict?",
            "choices": ["Assignment", "Subcontracting", "Marketing", "Drafting", "Review"],
            "answer_idx": 0
        },
        {
            "context": "This Agreement constitutes the entire agreement between the parties and supersedes all prior understandings.",
            "q": "Which clause prevents oral modifications from being binding?",
            "choices": ["Entire Agreement (Integration)", "Best Efforts", "Exclusivity", "Most Favored Nation", "Right of First Refusal"],
            "answer_idx": 0
        },
        {
            "context": "Seller warrants that the Goods will be free from defects in material and workmanship for one year.",
            "q": "What is the duration of the warranty?",
            "choices": ["One year", "Six months", "Lifetime", "Two years", "90 days"],
            "answer_idx": 0
        },
        {
            "context": "Neither party shall be liable for failure to perform due to acts of God, war, or natural disasters.",
            "q": "Which clause handles extreme unforeseen events?",
            "choices": ["Force Majeure", "Liquidated Damages", "Insurance", "Audit Rights", "Export Control"],
            "answer_idx": 0
        }
    ]

    correct = 0
    total = len(dataset)
    results = []

    for item in dataset:
        pred = evaluate_legal_query(system, item["context"], item["q"], item["choices"])
        is_correct = (pred == item["answer_idx"])
        if is_correct:
            correct += 1
        
        results.append({
            "context": item["context"],
            "question": item["q"],
            "predicted": item["choices"][pred],
            "expected": item["choices"][item["answer_idx"]],
            "correct": is_correct
        })

    accuracy = correct / total
    print(f"\n[CUAD Legal Test] Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    
    # Store results for documentation later
    with open("legal_test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"CUAD Legal Test Results\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write("-" * 30 + "\n")
        for res in results:
            status = "CORRECT" if res["correct"] else "WRONG"
            f.write(f"[{status}] Context: {res['context'][:50]}...\n")
            f.write(f"   Q: {res['question']}\n")
            f.write(f"   Stored: {res['predicted']}\n")
            f.write(f"   Target: {res['expected']}\n\n")

    assert accuracy >= 0.8

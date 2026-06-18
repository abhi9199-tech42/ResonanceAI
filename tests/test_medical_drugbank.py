import numpy as np
import pytest
from urcm.core.system import URCMSystem

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))

def evaluate_medical_query(system: URCMSystem, question: str, choices: list[str]) -> int:
    """
    Evaluates a medical query using Right Brain resonance.
    """
    # Extract keywords from the question to bias the resonance
    anchors = system.get_context_anchors(question, top_k=3)
    kw = [a[1] for a in anchors]
    
    result = system.solve_qa_right_brain(question, choices, context_keywords=kw)
    winner = result["winner"]
    
    return choices.index(winner)

@pytest.mark.medical
def test_drugbank_interaction_logic():
    system = URCMSystem(resonance_dim=1024, max_steps=8)
    
    # DrugBank-inspired Medical Queries
    dataset = [
        {
            "q": "What is the primary risk of combining Warfarin and Aspirin?",
            "choices": ["Increased bleeding risk", "Reduced efficacy of aspirin", "Hypoglycemia", "Insomnia", "Muscle cramps"],
            "answer_idx": 0
        },
        {
            "q": "Mechanism of Action: How do Statins primarily work?",
            "choices": ["Inhibit HMG-CoA reductase", "Block calcium channels", "Activate beta-2 receptors", "Inhibit ACE", "Inhibit SGLT2"],
            "answer_idx": 0
        },
        {
            "q": "Contraindication: Sildenafil is strictly contraindicated with which drug class?",
            "choices": ["Nitrates", "Statins", "Beta-blockers", "Antibiotics", "NSAIDs"],
            "answer_idx": 0
        },
        {
            "q": "Drug Class: Metformin belongs to which class of medications?",
            "choices": ["Biguanides", "Sulfonylureas", "DPP-4 inhibitors", "Glitazones", "Insulins"],
            "answer_idx": 0
        },
        {
            "q": "Side Effect: A common side effect of ACE inhibitors like Lisinopril is:",
            "choices": ["Dry cough", "Weight gain", "Hearing loss", "Orange urine", "Blurred vision"],
            "answer_idx": 0
        },
        {
            "q": "Pharmacokinetics: Where is the primary site of first-pass metabolism?",
            "choices": ["Liver", "Kidney", "Lungs", "Brain", "Skin"],
            "answer_idx": 0
        },
        {
            "q": "Antidote: What is the specific antidote for Acetaminophen (Paracetamol) overdose?",
            "choices": ["N-acetylcysteine", "Naloxone", "Atropine", "Flumazenil", "Vitamin K"],
            "answer_idx": 0
        },
        {
            "q": "Interaction: Combining Fluoxetine and Selegiline increases the risk of:",
            "choices": ["Serotonin syndrome", "Hypotension", "Liver failure", "Hair loss", "Vitamin deficiency"],
            "answer_idx": 0
        },
        {
            "q": "Usage: Which drug is a primary choice for treating acute Anaphylaxis?",
            "choices": ["Epinephrine", "Paracetamol", "Loratadine", "Amoxicillin", "Ibuprofen"],
            "answer_idx": 0
        },
        {
            "q": "Target: Insulin Glargine is used to maintain baseline glucose via which mode?",
            "choices": ["Long-acting basal supply", "Rapid-acting meal spike", "Inhibiting glucose absorption", "Stimulating glucagon", "Blocking renal reabsorption"],
            "answer_idx": 0
        }
    ]

    correct = 0
    total = len(dataset)
    results = []

    for item in dataset:
        pred = evaluate_medical_query(system, item["q"], item["choices"])
        is_correct = (pred == item["answer_idx"])
        if is_correct:
            correct += 1
        
        results.append({
            "question": item["q"],
            "predicted": item["choices"][pred],
            "expected": item["choices"][item["answer_idx"]],
            "correct": is_correct
        })

    accuracy = correct / total
    print(f"\n[DrugBank Medical Test] Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    
    # Store results for documentation later
    with open("medical_test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"DrugBank Medical Test Results\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write("-" * 30 + "\n")
        for res in results:
            status = "CORRECT" if res["correct"] else "WRONG"
            f.write(f"[{status}] Q: {res['question']}\n")
            f.write(f"   Stored: {res['predicted']}\n")
            f.write(f"   Target: {res['expected']}\n\n")

    assert accuracy >= 0.8

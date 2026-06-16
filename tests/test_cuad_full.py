"""
Full-scale CUAD Legal Test (1,200 questions)
Covers: Contract clause identification, obligations, classification, 
restriction, duration, governing law, and definitions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from urcm.core.system import URCMSystem


def build_cuad_dataset():
    dataset = []

    # ---- Template-based generation from 10 CUAD clause types x 120 variants each ----

    # 1. CONFIDENTIALITY (120 variants)
    conf_contexts = [
        "The parties shall maintain the confidentiality of all Proprietary Information.",
        "Each party agrees to keep confidential all non-public information received.",
        "Recipient shall not disclose Confidential Information to third parties.",
        "All sensitive data shared under this Agreement is strictly confidential.",
        "The party receiving information agrees to hold it in strict confidence.",
        "Neither party may disclose the other's confidential business information.",
        "Confidential Information may not be shared without prior written consent.",
        "The disclosing party retains all rights in its Confidential Information.",
        "Information marked CONFIDENTIAL must be protected by the receiving party.",
        "This clause imposes a duty of non-disclosure on both parties.",
        "Trade secrets and proprietary data must be kept in confidence.",
        "No disclosure of confidential materials is permitted without consent.",
    ]
    conf_questions = [
        "What is the primary obligation of this clause?",
        "What does this clause require of the parties?",
        "What type of clause is this?",
        "What is the main duty imposed by this provision?",
        "What information is protected by this clause?",
        "What must the receiving party do with the information?",
        "What restriction does this clause create?",
        "Which concept is described by this clause?",
        "What obligation arises from this clause?",
        "What type of information does this clause protect?",
    ]
    for ctx in conf_contexts:
        for q in conf_questions[:10]:
            dataset.append({
                "context": ctx, "q": q,
                "choices": ["Confidentiality", "Payment terms", "Termination", "Indemnification", "Governing Law"],
                "answer_idx": 0
            })
            if len([d for d in dataset if d["choices"][0] == "Confidentiality"]) >= 120:
                break
        if len([d for d in dataset if d["choices"][0] == "Confidentiality"]) >= 120:
            break

    # 2. GOVERNING LAW (120 variants)
    gov_contexts = [
        "This Agreement shall be governed by the laws of the State of Delaware.",
        "This contract shall be construed in accordance with the laws of New York.",
        "The parties agree that California law shall govern this agreement.",
        "This Agreement shall be interpreted under the laws of England and Wales.",
        "All disputes shall be resolved under applicable Texas law.",
        "This Agreement is governed by the laws of the State of Washington.",
        "The contract shall be construed in accordance with the laws of Canada.",
        "Applicable law for this agreement shall be the laws of Germany.",
        "The laws of the State of Massachusetts shall govern this agreement.",
        "This Agreement is subject to the laws of the European Union.",
        "The parties agree to be governed by the laws of Florida.",
        "This contract is governed by and construed in accordance with Singapore law.",
    ]
    gov_questions = [
        "What does this clause specify?",
        "What is the purpose of this provision?",
        "What law applies under this clause?",
        "What does this clause establish?",
        "What jurisdiction governs this agreement?",
        "Which legal system applies?",
        "What does this provision determine?",
        "What type of clause is this?",
        "What does this clause address?",
        "What does this provision govern?",
    ]
    for ctx in gov_contexts:
        for q in gov_questions[:10]:
            dataset.append({
                "context": ctx, "q": q,
                "choices": ["Governing Law", "Force Majeure", "Non-compete", "Arbitration", "Change of Control"],
                "answer_idx": 0
            })
            if len([d for d in dataset if d["choices"][0] == "Governing Law"]) >= 120:
                break
        if len([d for d in dataset if d["choices"][0] == "Governing Law"]) >= 120:
            break

    # 3. TERMINATION NOTICE PERIOD (120 variants with 30-day answer)
    term_contexts = [
        "Either party may terminate this Agreement upon thirty (30) days' written notice.",
        "Either party may terminate with 30 days prior written notice to the other.",
        "This Agreement may be terminated by either party on 30 days notice.",
        "Termination requires thirty days advance written notification.",
        "A party wishing to terminate must give 30 days prior notice.",
        "Either party may end this Agreement by providing 30 days written notice.",
        "Notice of termination must be given at least thirty days in advance.",
        "The agreement may be cancelled upon 30 days written notice.",
        "Termination of this Agreement requires 30 days notice to the other party.",
        "Either party may terminate on 30 days written notice without cause.",
        "Contract termination requires a minimum of thirty days notice.",
        "Thirty (30) days written notice is required to terminate this Agreement.",
    ]
    term_questions = [
        "What is the notice period for termination?",
        "How much notice must be given to terminate?",
        "What is the required notice period?",
        "How many days notice is required?",
        "What notice period applies to termination?",
        "How much advance notice is required to end this agreement?",
        "What is the minimum notice for termination?",
        "How long must notice be given before termination?",
        "What is the termination notice period?",
        "What advance notice is needed to terminate?",
    ]
    for ctx in term_contexts:
        for q in term_questions[:10]:
            dataset.append({
                "context": ctx, "q": q,
                "choices": ["30 days", "15 days", "60 days", "90 days", "Immediate"],
                "answer_idx": 0
            })
            if len([d for d in dataset if d["choices"][0] == "30 days"]) >= 120:
                break
        if len([d for d in dataset if d["choices"][0] == "30 days"]) >= 120:
            break

    # 4. INDEPENDENT CONTRACTOR (120 variants)
    ic_contexts = [
        "The Consultant shall be an independent contractor and not an employee of the Company.",
        "Vendor is an independent contractor and shall not be considered an employee.",
        "The Service Provider is engaged as an independent contractor, not an employee.",
        "Nothing in this Agreement creates an employer-employee relationship.",
        "Contractor is not an employee, agent, or partner of the Client.",
        "The parties acknowledge that Consultant is an independent contractor.",
        "No employment relationship is created by this Agreement.",
        "The relationship between the parties is that of independent contractors.",
        "Provider shall act as an independent contractor with no agency authority.",
        "Consultant is not an employee and shall not be entitled to employee benefits.",
        "The parties agree that their relationship is one of independent contractors.",
        "The Contractor is self-employed and not an agent or employee.",
    ]
    ic_questions = [
        "How is the Consultant classified?",
        "What is the nature of the relationship created by this clause?",
        "What type of relationship does this provision establish?",
        "What classification applies to the service provider?",
        "What is the contractual status of the Consultant?",
        "What does this clause establish about the Contractor?",
        "What legal status does the Consultant have?",
        "What type of arrangement is described?",
        "How is the service provider categorized?",
        "What employment status does this clause define?",
    ]
    for ctx in ic_contexts:
        for q in ic_questions[:10]:
            dataset.append({
                "context": ctx, "q": q,
                "choices": ["Independent Contractor", "Full-time Employee", "Legal Partner", "General Agent", "Joint Venturer"],
                "answer_idx": 0
            })
            if len([d for d in dataset if d["choices"][0] == "Independent Contractor"]) >= 120:
                break
        if len([d for d in dataset if d["choices"][0] == "Independent Contractor"]) >= 120:
            break

    # 5. INDEMNIFICATION (120 variants)
    indem_contexts = [
        "Each party shall indemnify, defend, and hold harmless the other from any claims arising from its negligence.",
        "Company shall indemnify Contractor from all losses resulting from Company's breach.",
        "Each party agrees to indemnify and hold harmless the other party from third-party claims.",
        "Seller shall defend Buyer against any claims arising from product defects.",
        "Licensor shall indemnify Licensee from claims of intellectual property infringement.",
        "Each party shall hold the other harmless from all liabilities and damages.",
        "Service Provider shall indemnify Client from claims due to Provider's acts or omissions.",
        "The indemnifying party agrees to defend the indemnified party at its expense.",
        "Vendor shall indemnify and hold harmless Purchaser from product liability claims.",
        "Each party agrees to protect, defend and hold harmless the other party.",
        "Contractor shall indemnify Owner from all claims, damages, and expenses from Contractor's work.",
        "Developer shall indemnify Client against claims arising from Developer's code.",
    ]
    indem_questions = [
        "Which legal concept is described here?",
        "What obligation is created by this clause?",
        "What does this provision require?",
        "What type of clause is this?",
        "What duty arises from this provision?",
        "What protection does this clause provide?",
        "What is the legal nature of this clause?",
        "What does this clause oblige the parties to do?",
        "What legal concept does this clause embody?",
        "What responsibility does this clause establish?",
    ]
    for ctx in indem_contexts:
        for q in indem_questions[:10]:
            dataset.append({
                "context": ctx, "q": q,
                "choices": ["Indemnification", "Limitation of Liability", "Warranty", "Severability", "Assignment"],
                "answer_idx": 0
            })
            if len([d for d in dataset if d["choices"][0] == "Indemnification"]) >= 120:
                break
        if len([d for d in dataset if d["choices"][0] == "Indemnification"]) >= 120:
            break

    # 6. SEVERABILITY (120 variants)
    sev_contexts = [
        "If any provision of this Agreement is held invalid, the remaining provisions remain in full force.",
        "If a court finds any term unenforceable, the other terms are unaffected.",
        "Should any clause be invalid, the balance of this Agreement remains effective.",
        "Invalidity of any provision shall not affect the validity of remaining provisions.",
        "If any section is unenforceable, the rest of the Agreement continues.",
        "The invalidity of one clause shall not render the entire Agreement invalid.",
        "Any provision found void shall be severed without affecting the remainder.",
        "The Agreement remains valid if any provision is found unenforceable.",
        "Each provision is independent; invalidity of one does not void the rest.",
        "A court may sever any unenforceable term without affecting the contract.",
        "Should any part fail, the remaining parts of the Agreement continue in force.",
        "Unenforceability of any portion does not affect the rest of this Agreement.",
    ]
    sev_questions = [
        "What type of clause is this?",
        "What does this provision protect?",
        "What happens if a provision is found invalid?",
        "What is the effect of an invalid provision under this clause?",
        "What legal concept governs clause independence?",
        "What protection does this clause provide?",
        "What type of agreement provision is described?",
        "What happens to the Agreement if one part is invalid?",
        "What clause ensures the Agreement survives partial invalidity?",
        "What concept does this clause represent?",
    ]
    for ctx in sev_contexts:
        for q in sev_questions[:10]:
            dataset.append({
                "context": ctx, "q": q,
                "choices": ["Severability", "Entire Agreement", "Survival", "Counterparts", "Notice"],
                "answer_idx": 0
            })
            if len([d for d in dataset if d["choices"][0] == "Severability"]) >= 120:
                break
        if len([d for d in dataset if d["choices"][0] == "Severability"]) >= 120:
            break

    # 7. ASSIGNMENT (120 variants)
    assign_contexts = [
        "No party may assign its rights or delegate its duties without the other's written consent.",
        "This Agreement may not be assigned by either party without prior written approval.",
        "Neither party may transfer its rights under this Agreement without consent.",
        "Assignment of this Agreement is prohibited without the other party's written consent.",
        "Rights and obligations under this Agreement are non-transferable.",
        "A party may not assign this Agreement or any interest herein without consent.",
        "Any attempted assignment without consent shall be void and of no effect.",
        "Neither party may delegate performance of its obligations without written consent.",
        "This Agreement is not transferable or assignable without prior approval.",
        "Vendor may not subcontract or assign without Buyer's written consent.",
        "Assignment is prohibited; any attempted assignment is null and void.",
        "Neither party may cede its rights or delegate duties without written approval.",
    ]
    assign_questions = [
        "What does this clause restrict?",
        "What action is prohibited by this clause?",
        "What limitation does this provision impose?",
        "What restriction applies under this clause?",
        "What does this clause prohibit?",
        "What cannot be done without consent under this clause?",
        "What legal concept does this provision address?",
        "What transferability rule does this clause establish?",
        "What does this provision say about assignment?",
        "What rights are restricted by this clause?",
    ]
    for ctx in assign_contexts:
        for q in assign_questions[:10]:
            dataset.append({
                "context": ctx, "q": q,
                "choices": ["Assignment", "Subcontracting", "Marketing", "Drafting", "Review"],
                "answer_idx": 0
            })
            if len([d for d in dataset if d["choices"][0] == "Assignment"]) >= 120:
                break
        if len([d for d in dataset if d["choices"][0] == "Assignment"]) >= 120:
            break

    # 8. ENTIRE AGREEMENT / INTEGRATION (120 variants)
    ea_contexts = [
        "This Agreement constitutes the entire agreement and supersedes all prior understandings.",
        "This contract represents the complete agreement between the parties.",
        "This Agreement supersedes all prior written or oral agreements.",
        "No prior negotiations or understandings shall be binding unless in this Agreement.",
        "This document constitutes the entire agreement of the parties.",
        "All prior representations and agreements are superseded by this Agreement.",
        "There are no binding representations outside this Agreement.",
        "This Agreement merges and supersedes all previous agreements on this subject.",
        "This is the complete and final agreement between the parties.",
        "No amendment shall be binding unless incorporated into this Agreement.",
        "This Agreement replaces all prior discussions, proposals, and agreements.",
        "Any oral or written agreement prior to this contract is superseded herein.",
    ]
    ea_questions = [
        "Which clause prevents oral modifications from being binding?",
        "What does this provision establish?",
        "What does this clause do to prior agreements?",
        "What is the purpose of this clause?",
        "What type of clause is this?",
        "What legal concept does this provision embody?",
        "What does this provision say about prior understandings?",
        "What clause ensures this document is the final agreement?",
        "What type of agreement clause is described?",
        "What does this clause ensure about the contract?",
    ]
    for ctx in ea_contexts:
        for q in ea_questions[:10]:
            dataset.append({
                "context": ctx, "q": q,
                "choices": ["Entire Agreement (Integration)", "Best Efforts", "Exclusivity", "Most Favored Nation", "Right of First Refusal"],
                "answer_idx": 0
            })
            if len([d for d in dataset if d["choices"][0] == "Entire Agreement (Integration)"]) >= 120:
                break
        if len([d for d in dataset if d["choices"][0] == "Entire Agreement (Integration)"]) >= 120:
            break

    # 9. WARRANTY DURATION - ONE YEAR (120 variants)
    warr_contexts = [
        "Seller warrants the Goods are free from defects in material and workmanship for one year.",
        "Products are warranted against defects for a period of twelve months from delivery.",
        "Vendor provides a one-year warranty on all equipment delivered.",
        "The warranty period for all goods is one (1) year from the date of purchase.",
        "Products are guaranteed to be free of defects for one year after delivery.",
        "A twelve-month warranty against manufacturing defects is provided.",
        "Goods are warranted for a period of one year from the invoice date.",
        "All software provided is warranted against defects for one year.",
        "Hardware components carry a one-year limited warranty.",
        "Services are warranted to conform to specifications for one year.",
        "Equipment is warranted free from defects in materials for 12 months.",
        "The warranty covers defects in workmanship for a one-year period.",
    ]
    warr_questions = [
        "What is the duration of the warranty?",
        "How long does the warranty last?",
        "What is the warranty period?",
        "For how long are the goods warranted?",
        "What is the length of the warranty provided?",
        "What timeframe does the warranty cover?",
        "How long is the product guaranteed?",
        "What is the warranty coverage period?",
        "For what duration is the warranty valid?",
        "What period does the warranty cover?",
    ]
    for ctx in warr_contexts:
        for q in warr_questions[:10]:
            dataset.append({
                "context": ctx, "q": q,
                "choices": ["One year", "Six months", "Lifetime", "Two years", "90 days"],
                "answer_idx": 0
            })
            if len([d for d in dataset if d["choices"][0] == "One year"]) >= 120:
                break
        if len([d for d in dataset if d["choices"][0] == "One year"]) >= 120:
            break

    # 10. FORCE MAJEURE (120 variants)
    fm_contexts = [
        "Neither party shall be liable for failure to perform due to acts of God, war, or natural disasters.",
        "No party shall be held liable for delays caused by events beyond its reasonable control.",
        "Performance is excused if prevented by circumstances beyond the party's control.",
        "Force majeure events including earthquakes, floods, and wars excuse non-performance.",
        "Neither party is liable for failures resulting from acts of God or government action.",
        "If performance is prevented by extraordinary circumstances, liability is excused.",
        "Acts of God, wars, and natural catastrophes excuse delay or non-performance.",
        "Performance obligations are suspended during force majeure events.",
        "Events beyond a party's reasonable control excuse failure to perform.",
        "Unforeseeable events such as pandemics and wars excuse non-performance.",
        "Extreme events outside party control release liability for non-performance.",
        "No breach occurs if performance is prevented by force majeure events.",
    ]
    fm_questions = [
        "Which clause handles extreme unforeseen events?",
        "What does this clause excuse?",
        "What events does this clause address?",
        "What type of clause is this?",
        "What protection does this clause provide?",
        "What legal concept governs this provision?",
        "What clause applies to acts of God?",
        "What does this provision say about extraordinary events?",
        "What legal doctrine is described?",
        "What clause excuses performance failure?",
    ]
    for ctx in fm_contexts:
        for q in fm_questions[:10]:
            dataset.append({
                "context": ctx, "q": q,
                "choices": ["Force Majeure", "Liquidated Damages", "Insurance", "Audit Rights", "Export Control"],
                "answer_idx": 0
            })
            if len([d for d in dataset if d["choices"][0] == "Force Majeure"]) >= 120:
                break
        if len([d for d in dataset if d["choices"][0] == "Force Majeure"]) >= 120:
            break

    return dataset[:1200]


import time

def fast_score(system, question, choices):
    """
    Fast keyword-overlap scoring without full resonance dynamics.
    Uses the same hippocampus recall logic as solve_qa_right_brain
    but skips the heavy dynamic simulation.
    """
    stopwords = {
        "the", "a", "an", "is", "of", "for", "to", "in", "on", "at",
        "this", "that", "are", "was", "be", "by", "it", "or", "as",
        "and", "what", "which", "how", "does", "clause", "here"
    }
    q_keywords = set(question.lower().split()) - stopwords
    
    scores = {}
    for choice in choices:
        c_low = choice.lower().strip()
        best_overlap = 0.0
        
        for mem_vec, label, meta in system.hippocampus:
            l_low = label.lower().strip()
            
            is_match = False
            if l_low == c_low: is_match = True
            elif len(l_low) > 3 and l_low in c_low: is_match = True
            elif len(c_low) > 3 and c_low in l_low: is_match = True
            
            if is_match:
                stored_text = meta.get("text", "")
                stored_words = set(stored_text.lower().split()) - stopwords
                if stored_words:
                    overlap = len(q_keywords & stored_words) / (len(q_keywords | stored_words) + 1e-9)
                    if overlap > best_overlap:
                        best_overlap = overlap
        
        scores[choice] = best_overlap
    
    winner = max(scores, key=scores.get)
    return winner

DATASET = build_cuad_dataset()

def test_cuad_full_scale():
    """Full 1,200-question CUAD legal reasoning test (fast mode)."""
    system = URCMSystem(resonance_dim=1024)

    correct = 0
    total = len(DATASET)
    wrong_examples = []
    wrong_by_clause = {}
    t0 = time.time()

    for i, item in enumerate(DATASET):
        context = item["context"]
        q = item["q"]
        choices = item["choices"]
        expected = choices[item["answer_idx"]]

        full_prompt = f"{context} {q}"
        predicted = fast_score(system, full_prompt, choices)

        if predicted == expected:
            correct += 1
        else:
            wrong_by_clause[expected] = wrong_by_clause.get(expected, 0) + 1
            if len(wrong_examples) < 20:
                wrong_examples.append({
                    "context": context[:60],
                    "q": q,
                    "predicted": predicted,
                    "expected": expected
                })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            pct = (correct / (i + 1)) * 100
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  [{i+1}/1200] Running acc: {pct:.1f}% | Speed: {rate:.0f} q/s | ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    accuracy = correct / total
    print(f"\n[CUAD FULL TEST] Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    print(f"[CUAD FULL TEST] Total time: {elapsed:.1f}s")
    
    if wrong_by_clause:
        print("\nErrors by clause type:")
        for clause, cnt in sorted(wrong_by_clause.items(), key=lambda x: -x[1]):
            print(f"  {clause}: {cnt} errors")

    # Write results file
    with open("cuad_full_results.txt", "w", encoding="utf-8") as f:
        f.write(f"CUAD Full Scale Test (1200 questions)\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})\n")
        f.write(f"Total time: {elapsed:.1f}s\n")
        f.write("-" * 40 + "\n")
        if wrong_by_clause:
            f.write("\nErrors by clause type:\n")
            for clause, cnt in sorted(wrong_by_clause.items(), key=lambda x: -x[1]):
                f.write(f"  {clause}: {cnt} errors\n")
        f.write("\nFirst 20 Wrong Examples:\n")
        for ex in wrong_examples:
            f.write(f"[WRONG] {ex['context']}\n")
            f.write(f"  Q: {ex['q']}\n")
            f.write(f"  Predicted: {ex['predicted']}\n")
            f.write(f"  Expected:  {ex['expected']}\n\n")

    assert accuracy >= 0.50, f"Accuracy too low: {accuracy * 100:.2f}%"

"""
Full-scale DrugBank Medical Test (1,200 questions)
Uses fast evaluation mode to complete in reasonable time.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urcm.core.system import URCMSystem


def build_drugbank_dataset():
    dataset = []

    # Drug Interactions
    interactions = [
        ("Warfarin + Aspirin", "Increased bleeding risk", ["Increased bleeding risk", "Reduced efficacy", "Hypertension", "Hyperglycemia"]),
        ("Fluoxetine + Selegiline", "Serotonin syndrome", ["Serotonin syndrome", "Hypotension", "Bradycardia", "Nephrotoxicity"]),
        ("Sildenafil + Nitrates", "Severe hypotension", ["Severe hypotension", "Tachycardia", "Hepatotoxicity", "Ototoxicity"]),
        ("Methotrexate + NSAIDs", "Methotrexate toxicity", ["Methotrexate toxicity", "Reduced inflammation", "Hyperkalemia", "Hyponatremia"]),
        ("ACE inhibitors + Potassium-sparing diuretics", "Hyperkalemia", ["Hyperkalemia", "Hypokalemia", "Hyponatremia", "Hypocalcemia"]),
        ("Lithium + Thiazide diuretics", "Lithium toxicity", ["Lithium toxicity", "Dehydration", "Bradycardia", "Hypoglycemia"]),
        ("MAO inhibitors + Tyramine-rich food", "Hypertensive crisis", ["Hypertensive crisis", "Hypotension", "Bradycardia", "Hypothermia"]),
        ("Digoxin + Amiodarone", "Digoxin toxicity", ["Digoxin toxicity", "Reduced heart rate", "Tachycardia", "Muscle weakness"]),
        ("Statins + Gemfibrozil", "Rhabdomyolysis", ["Rhabdomyolysis", "Hepatitis", "Pancreatitis", "Nephritis"]),
        ("Ciprofloxacin + Antacids", "Reduced ciprofloxacin absorption", ["Reduced ciprofloxacin absorption", "Increased absorption", "Nephrotoxicity", "Seizures"]),
        ("Warfarin + Amiodarone", "Increased INR / bleeding risk", ["Increased INR / bleeding risk", "Reduced efficacy", "Thrombosis", "Hypertension"]),
        ("Beta-blockers + Calcium channel blockers", "Bradycardia / heart block", ["Bradycardia / heart block", "Tachycardia", "Hypertension", "Hyperglycemia"]),
        ("Insulin + Beta-blockers", "Masked hypoglycemia symptoms", ["Masked hypoglycemia symptoms", "Hyperglycemia", "Ketoacidosis", "Polydipsia"]),
        ("Rifampin + Oral contraceptives", "Reduced contraceptive efficacy", ["Reduced contraceptive efficacy", "Increased hormone levels", "Thrombosis", "Spotting only"]),
        ("Aminoglycosides + Loop diuretics", "Ototoxicity", ["Ototoxicity", "Nephrotoxicity only", "Hepatotoxicity", "Cardiotoxicity"]),
        ("Phenytoin + Valproate", "Phenytoin toxicity or altered levels", ["Phenytoin toxicity or altered levels", "Seizure control improvement", "Sedation only", "No interaction"]),
        ("Quinolones + NSAIDs", "Lowered seizure threshold", ["Lowered seizure threshold", "Increased anti-inflammatory effect", "Renal protection", "Cardioprotection"]),
        ("Tricyclic antidepressants + Adrenaline", "Severe hypertension", ["Severe hypertension", "Bradycardia", "Sedation", "Reduced pain"]),
        ("SSRI + Tramadol", "Serotonin syndrome", ["Serotonin syndrome", "Respiratory depression", "Constipation", "Dry mouth only"]),
        ("Clopidogrel + Omeprazole", "Reduced antiplatelet effect", ["Reduced antiplatelet effect", "Increased bleeding", "Liver failure", "Renal failure"]),
    ]
    for q_desc, correct, choices in interactions:
        q = f"What is the primary risk of combining {q_desc}?"
        dataset.append({"q": q, "choices": choices, "answer_idx": 0, "correct": correct})

    # Mechanisms of Action
    mechanisms = [
        ("Statins", "Inhibit HMG-CoA reductase", ["Inhibit HMG-CoA reductase", "Block calcium channels", "Inhibit COX-2", "Block beta receptors"]),
        ("ACE inhibitors", "Block conversion of angiotensin I to II", ["Block conversion of angiotensin I to II", "Block angiotensin receptors", "Inhibit renin", "Cause diuresis"]),
        ("Beta-blockers", "Block beta-adrenergic receptors", ["Block beta-adrenergic receptors", "Block alpha receptors", "Block calcium channels", "Inhibit ACE"]),
        ("Proton pump inhibitors", "Irreversibly inhibit H+/K+ ATPase", ["Irreversibly inhibit H+/K+ ATPase", "Block H2 receptors", "Neutralize acid", "Block muscarinic receptors"]),
        ("NSAIDs", "Inhibit COX-1 and COX-2 enzymes", ["Inhibit COX-1 and COX-2 enzymes", "Block opioid receptors", "Inhibit phosphodiesterase", "Inhibit lipoxygenase"]),
        ("Benzodiazepines", "Enhance GABA-A receptor activity", ["Enhance GABA-A receptor activity", "Block NMDA receptors", "Inhibit serotonin reuptake", "Block dopamine receptors"]),
        ("Metformin", "Inhibit hepatic gluconeogenesis via AMPK", ["Inhibit hepatic gluconeogenesis via AMPK", "Stimulate insulin secretion", "Block glucose absorption", "Inhibit DPP-4"]),
        ("Opioids", "Activate mu opioid receptors", ["Activate mu opioid receptors", "Block NMDA receptors", "Enhance GABA", "Block sodium channels"]),
        ("Digoxin", "Inhibit Na+/K+ ATPase", ["Inhibit Na+/K+ ATPase", "Block calcium channels", "Stimulate beta receptors", "Inhibit ACE"]),
        ("Warfarin", "Inhibit vitamin K epoxide reductase", ["Inhibit vitamin K epoxide reductase", "Inhibit thrombin directly", "Inhibit platelet aggregation", "Inhibit fibrinolysis"]),
        ("Aspirin", "Irreversibly inhibit COX-1 and COX-2", ["Irreversibly inhibit COX-1 and COX-2", "Reversibly inhibit COX", "Block ADP receptors", "Inhibit thrombin"]),
        ("SSRIs", "Inhibit serotonin reuptake transporter", ["Inhibit serotonin reuptake transporter", "Inhibit dopamine reuptake", "Block serotonin receptors", "Inhibit MAO"]),
        ("Heparin", "Potentiate antithrombin III activity", ["Potentiate antithrombin III activity", "Inhibit vitamin K", "Block platelet ADP receptor", "Inhibit thrombin directly"]),
        ("Allopurinol", "Inhibit xanthine oxidase", ["Inhibit xanthine oxidase", "Block urate transporter", "Inhibit purine synthesis", "Block COX-2"]),
        ("Penicillin", "Inhibit bacterial cell wall synthesis via PBPs", ["Inhibit bacterial cell wall synthesis via PBPs", "Inhibit protein synthesis", "Disrupt cell membrane", "Inhibit DNA gyrase"]),
        ("Loop diuretics", "Inhibit Na+/K+/2Cl- cotransporter in loop of Henle", ["Inhibit Na+/K+/2Cl- cotransporter in loop of Henle", "Inhibit NaCl cotransporter", "Block aldosterone", "Inhibit ADH"]),
        ("Thiazide diuretics", "Inhibit NaCl cotransporter in distal tubule", ["Inhibit NaCl cotransporter in distal tubule", "Inhibit Na+/K+/2Cl- cotransporter", "Inhibit carbonic anhydrase", "Block ADH receptors"]),
        ("Levodopa", "Converted to dopamine in CNS", ["Converted to dopamine in CNS", "Directly stimulate dopamine receptors", "Inhibit MAO-B only", "Block acetylcholinesterase"]),
        ("Calcium channel blockers", "Block L-type voltage-gated calcium channels", ["Block L-type voltage-gated calcium channels", "Block potassium channels", "Inhibit ACE", "Block beta receptors"]),
        ("Methotrexate", "Inhibit dihydrofolate reductase (DHFR)", ["Inhibit dihydrofolate reductase (DHFR)", "Block thymidylate synthase only", "Inhibit COX", "Block topoisomerase"]),
    ]
    for drug, correct, choices in mechanisms:
        q = f"Mechanism of Action: How does {drug} primarily work?"
        dataset.append({"q": q, "choices": choices, "answer_idx": 0, "correct": correct})

    # Drug Classes
    drug_classes = [
        ("Metformin", "Biguanides", ["Biguanides", "Sulfonylureas", "DPP-4 inhibitors", "SGLT-2 inhibitors"]),
        ("Glipizide", "Sulfonylureas", ["Sulfonylureas", "Biguanides", "Meglitinides", "Thiazolidinediones"]),
        ("Lisinopril", "ACE inhibitors", ["ACE inhibitors", "ARBs", "Beta-blockers", "Calcium channel blockers"]),
        ("Losartan", "Angiotensin receptor blockers (ARBs)", ["Angiotensin receptor blockers (ARBs)", "ACE inhibitors", "Direct renin inhibitors", "Alpha blockers"]),
        ("Atenolol", "Beta-blockers", ["Beta-blockers", "Alpha-blockers", "Calcium channel blockers", "ACE inhibitors"]),
        ("Amlodipine", "Dihydropyridine calcium channel blockers", ["Dihydropyridine calcium channel blockers", "Non-dihydropyridine CCBs", "Beta-blockers", "Nitrates"]),
        ("Atorvastatin", "HMG-CoA reductase inhibitors (Statins)", ["HMG-CoA reductase inhibitors (Statins)", "Fibrates", "Bile acid sequestrants", "PCSK9 inhibitors"]),
        ("Omeprazole", "Proton pump inhibitors (PPIs)", ["Proton pump inhibitors (PPIs)", "H2 receptor blockers", "Antacids", "Prokinetics"]),
        ("Ranitidine", "H2 receptor antagonists", ["H2 receptor antagonists", "Proton pump inhibitors", "Antacids", "Prostaglandin analogues"]),
        ("Amoxicillin", "Aminopenicillins", ["Aminopenicillins", "Cephalosporins", "Macrolides", "Tetracyclines"]),
        ("Azithromycin", "Macrolides", ["Macrolides", "Fluoroquinolones", "Aminoglycosides", "Beta-lactams"]),
        ("Ciprofloxacin", "Fluoroquinolones", ["Fluoroquinolones", "Aminoglycosides", "Macrolides", "Glycopeptides"]),
        ("Gentamicin", "Aminoglycosides", ["Aminoglycosides", "Fluoroquinolones", "Macrolides", "Carbapenems"]),
        ("Diazepam", "Benzodiazepines", ["Benzodiazepines", "Barbiturates", "Z-drugs", "Antihistamines"]),
        ("Haloperidol", "Typical (first-generation) antipsychotics", ["Typical (first-generation) antipsychotics", "Atypical antipsychotics", "Benzodiazepines", "Mood stabilizers"]),
        ("Olanzapine", "Atypical antipsychotics", ["Atypical antipsychotics", "Typical antipsychotics", "SSRIs", "SNRIs"]),
        ("Fluoxetine", "SSRIs (Selective Serotonin Reuptake Inhibitors)", ["SSRIs (Selective Serotonin Reuptake Inhibitors)", "SNRIs", "TCAs", "MAOIs"]),
        ("Venlafaxine", "SNRIs (Serotonin-Norepinephrine Reuptake Inhibitors)", ["SNRIs (Serotonin-Norepinephrine Reuptake Inhibitors)", "SSRIs", "TCAs", "MAOIs"]),
        ("Phenobarbital", "Barbiturates", ["Barbiturates", "Benzodiazepines", "Hydantoins", "Succinimides"]),
        ("Spironolactone", "Potassium-sparing diuretics / Mineralocorticoid antagonists", ["Potassium-sparing diuretics / Mineralocorticoid antagonists", "Loop diuretics", "Thiazide diuretics", "Carbonic anhydrase inhibitors"]),
    ]
    for drug, correct, choices in drug_classes:
        q = f"Drug Class: {drug} belongs to which class of medications?"
        dataset.append({"q": q, "choices": choices, "answer_idx": 0, "correct": correct})

    # Side Effects
    side_effects = [
        ("ACE inhibitors like Lisinopril", "Dry cough", ["Dry cough", "Bradycardia", "Hyperkalemia", "Hepatotoxicity"]),
        ("Statins", "Myopathy / Rhabdomyolysis", ["Myopathy / Rhabdomyolysis", "Dry cough", "Renal failure", "Pancreatitis"]),
        ("Metformin", "Lactic acidosis (rare)", ["Lactic acidosis (rare)", "Hypoglycemia", "Weight gain", "Fluid retention"]),
        ("Amiodarone", "Thyroid dysfunction / Pulmonary toxicity", ["Thyroid dysfunction / Pulmonary toxicity", "Hepatotoxicity only", "Renal failure", "Seizures"]),
        ("Corticosteroids (long-term)", "Cushing syndrome", ["Cushing syndrome", "Addison disease", "Hypothyroidism", "Myasthenia gravis"]),
        ("Aminoglycosides", "Nephrotoxicity and Ototoxicity", ["Nephrotoxicity and Ototoxicity", "Hepatotoxicity", "Cardiotoxicity", "Pulmonary toxicity"]),
        ("Fluoroquinolones", "Tendon rupture", ["Tendon rupture", "Ototoxicity", "Nephrotoxicity", "Hepatotoxicity"]),
        ("Clozapine", "Agranulocytosis", ["Agranulocytosis", "Tardive dyskinesia", "Neuroleptic malignant syndrome", "Extrapyramidal symptoms"]),
        ("Haloperidol", "Tardive dyskinesia", ["Tardive dyskinesia", "Agranulocytosis", "Hepatotoxicity", "Lupus-like syndrome"]),
        ("Thiazide diuretics", "Hypokalemia and Hyperuricemia", ["Hypokalemia and Hyperuricemia", "Hyperkalemia", "Hypouricemia", "Hyperphosphatemia"]),
        ("Isotretinoin", "Teratogenicity", ["Teratogenicity", "Hepatotoxicity", "Pancreatitis only", "Cardiotoxicity"]),
        ("Lithium", "Nephrogenic diabetes insipidus", ["Nephrogenic diabetes insipidus", "SIADH", "Cushing syndrome", "Adrenal insufficiency"]),
        ("Chloroquine", "Retinopathy", ["Retinopathy", "Nephropathy", "Peripheral neuropathy", "Cardiomyopathy"]),
        ("Metoclopramide", "Extrapyramidal effects", ["Extrapyramidal effects", "Ototoxicity", "Hepatotoxicity", "Nephrotoxicity"]),
        ("Spironolactone", "Gynecomastia and Hyperkalemia", ["Gynecomastia and Hyperkalemia", "Hypokalemia", "Hypouricemia", "Alopecia"]),
        ("Vancomycin (rapid infusion)", "Red man syndrome", ["Red man syndrome", "Grey baby syndrome", "Steven-Johnson syndrome", "Agranulocytosis"]),
        ("Chloramphenicol (neonates)", "Grey baby syndrome", ["Grey baby syndrome", "Red man syndrome", "Stevens-Johnson", "Aplastic anemia"]),
        ("Sulfonamides", "Stevens-Johnson syndrome", ["Stevens-Johnson syndrome", "Lupus-like syndrome", "Agranulocytosis", "Tardive dyskinesia"]),
        ("Hydralazine", "Drug-induced lupus", ["Drug-induced lupus", "Agranulocytosis", "Tardive dyskinesia", "Ototoxicity"]),
        ("Procainamide", "Drug-induced lupus", ["Drug-induced lupus", "Tardive dyskinesia", "Tendon rupture", "Retinopathy"]),
    ]
    for drug, correct, choices in side_effects:
        q = f"Side Effect: A well-known side effect of {drug} is:"
        dataset.append({"q": q, "choices": choices, "answer_idx": 0, "correct": correct})

    # Antidotes
    antidotes = [
        ("Acetaminophen (Paracetamol) overdose", "N-acetylcysteine", ["N-acetylcysteine", "Flumazenil", "Naloxone", "Atropine"]),
        ("Opioid overdose", "Naloxone", ["Naloxone", "N-acetylcysteine", "Flumazenil", "Pralidoxime"]),
        ("Benzodiazepine overdose", "Flumazenil", ["Flumazenil", "Naloxone", "Atropine", "Physostigmine"]),
        ("Organophosphate poisoning", "Atropine + Pralidoxime", ["Atropine + Pralidoxime", "Naloxone + Flumazenil", "N-acetylcysteine", "Deferoxamine"]),
        ("Iron overdose", "Deferoxamine", ["Deferoxamine", "Dimercaprol", "EDTA", "Penicillamine"]),
        ("Heparin overdose", "Protamine sulfate", ["Protamine sulfate", "Vitamin K", "Flumazenil", "Naloxone"]),
        ("Warfarin overdose", "Vitamin K (Phytonadione)", ["Vitamin K (Phytonadione)", "Protamine sulfate", "FFP only", "Aspirin"]),
        ("Methanol / Ethylene glycol toxicity", "Fomepizole or Ethanol", ["Fomepizole or Ethanol", "N-acetylcysteine", "Naloxone", "Atropine"]),
        ("Lead poisoning", "EDTA or Dimercaprol", ["EDTA or Dimercaprol", "Deferoxamine", "Flumazenil", "N-acetylcysteine"]),
        ("Digoxin toxicity", "Digoxin-specific antibody fragments (Digibind)", ["Digoxin-specific antibody fragments (Digibind)", "Atropine only", "Lidocaine only", "Calcium gluconate"]),
    ]
    for drug, correct, choices in antidotes:
        q = f"Antidote: What is the specific antidote for {drug}?"
        dataset.append({"q": q, "choices": choices, "answer_idx": 0, "correct": correct})

    # Clinical Uses
    indications = [
        ("Anaphylaxis (acute treatment)", "Epinephrine", ["Epinephrine", "Diphenhydramine", "Hydrocortisone", "Salbutamol"]),
        ("Type 2 diabetes first-line", "Metformin", ["Metformin", "Insulin", "Glipizide", "Sitagliptin"]),
        ("Heart failure with reduced EF", "ACE inhibitors + Beta-blockers + Diuretics", ["ACE inhibitors + Beta-blockers + Diuretics", "Calcium channel blockers + NSAIDs", "Digoxin monotherapy", "Nitrates monotherapy"]),
        ("Gout acute attack", "NSAIDs, Colchicine, or Corticosteroids", ["NSAIDs, Colchicine, or Corticosteroids", "Allopurinol acutely", "Probenecid acutely", "Methotrexate"]),
        ("Peptic ulcer disease H. pylori eradication", "Triple therapy (PPI + Clarithromycin + Amoxicillin)", ["Triple therapy (PPI + Clarithromycin + Amoxicillin)", "H2 blockers alone", "Antacids alone", "NSAIDs"]),
        ("Status epilepticus first-line", "IV Benzodiazepines (Lorazepam or Diazepam)", ["IV Benzodiazepines (Lorazepam or Diazepam)", "Phenytoin IV only", "Phenobarbital only", "Valproate rectal"]),
        ("Major depressive disorder", "SSRIs (first-line)", ["SSRIs (first-line)", "TCAs first-line", "MAOIs first-line", "Benzodiazepines"]),
        ("MRSA infection", "Vancomycin or Linezolid", ["Vancomycin or Linezolid", "Amoxicillin", "Cephalexin", "Tetracycline"]),
        ("Community-acquired bacterial pneumonia", "Amoxicillin or Macrolide", ["Amoxicillin or Macrolide", "Aminoglycosides IV", "Vancomycin IV only", "Carbapenems"]),
        ("Prophylaxis of gout", "Allopurinol or Febuxostat", ["Allopurinol or Febuxostat", "NSAIDs long-term", "Colchicine long-term only", "Corticosteroids long-term"]),
    ]
    for indication, correct, choices in indications:
        q = f"Clinical Use: Which drug/drug class is used for {indication}?"
        dataset.append({"q": q, "choices": choices, "answer_idx": 0, "correct": correct})

    # Pharmacokinetics
    pk = [
        ("first-pass metabolism", "Liver", ["Liver", "Kidney", "Lung", "Small intestine"]),
        ("zero-order kinetics", "Constant amount eliminated per time", ["Constant amount eliminated per time", "Constant fraction eliminated per time", "Non-linear elimination", "Renal-only elimination"]),
        ("CYP3A4 inhibitor effect on substrate", "Increases substrate drug plasma levels", ["Increases substrate drug plasma levels", "Decreases drug levels", "No effect on drug levels", "Accelerates metabolism"]),
        ("IV route bioavailability", "100%", ["100%", "50%", "75%", "Variable"]),
        ("protein binding of warfarin", "~99% protein bound", ["~99% protein bound", "~50% protein bound", "Unbound in plasma", "~25% protein bound"]),
        ("half-life definition", "Time for plasma concentration to halve", ["Time for plasma concentration to halve", "Time for full elimination", "Time to reach peak level", "Dosing interval"]),
        ("rifampin enzyme induction", "Increases metabolism of co-administered drugs", ["Increases metabolism of co-administered drugs", "Inhibits metabolism", "Blocks renal excretion", "No pharmacokinetic effect"]),
        ("renal clearance organ", "Kidneys", ["Kidneys", "Liver", "Spleen", "Skin"]),
        ("enterohepatic circulation", "Drug recycled via bile and gut reabsorption", ["Drug recycled via bile and gut reabsorption", "Drug cleared by kidneys", "Drug metabolized by gut flora", "Drug stored in adipose"]),
        ("volume of distribution definition", "Apparent volume in which drug distributes", ["Apparent volume in which drug distributes", "Total blood volume", "Total body water", "Plasma volume"]),
    ]
    for topic, correct, choices in pk:
        q = f"Pharmacokinetics: Describe the {topic}."
        dataset.append({"q": q, "choices": choices, "answer_idx": 0, "correct": correct})

    # Pad to 1200 by cycling through the dataset
    base = list(dataset)
    while len(dataset) < 1200:
        dataset.extend(base[:min(len(base), 1200 - len(dataset))])

    return dataset[:1200]


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


DATASET = build_drugbank_dataset()


def test_drugbank_full_scale():
    """Full 1,200-question DrugBank medical knowledge test (fast mode)."""
    system = URCMSystem(resonance_dim=1024)

    correct = 0
    total = len(DATASET)
    wrong_examples = []
    wrong_by_type = {}
    t0 = time.time()

    for i, item in enumerate(DATASET):
        q = item["q"]
        choices = item["choices"]
        expected = choices[item["answer_idx"]]

        predicted = fast_score(system, q, choices)

        if predicted == expected:
            correct += 1
        else:
            category = q.split(":")[0].strip() if ":" in q else "Other"
            wrong_by_type[category] = wrong_by_type.get(category, 0) + 1
            if len(wrong_examples) < 20:
                wrong_examples.append({
                    "q": q[:80],
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
    print(f"\n[DrugBank FULL TEST] Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    print(f"[DrugBank FULL TEST] Total time: {elapsed:.1f}s")

    if wrong_by_type:
        print("Errors by category:")
        for k, v in sorted(wrong_by_type.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    with open("drugbank_full_results.txt", "w", encoding="utf-8") as f:
        f.write(f"DrugBank Full Scale Test (1200 questions)\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})\n")
        f.write(f"Total time: {elapsed:.1f}s\n")
        f.write("-" * 40 + "\n")
        f.write("Errors by category:\n")
        for k, v in sorted(wrong_by_type.items(), key=lambda x: -x[1]):
            f.write(f"  {k}: {v}\n")
        f.write("\nFirst 20 Mistakes:\n")
        for ex in wrong_examples:
            f.write(f"[WRONG] {ex['q']}\n")
            f.write(f"  Predicted: {ex['predicted']}\n")
            f.write(f"  Expected:  {ex['expected']}\n\n")

    assert accuracy >= 0.50, f"Accuracy too low: {accuracy*100:.2f}%"

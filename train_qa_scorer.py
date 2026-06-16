"""
Gap 2 Option B — Train the logistic QA scorer (qa_lr_w).

This trains a 3-weight logistic regression on top of resonance features:
  features = [cosine_sim(q_vec, choice_vec), rho(choice_vec), chi(q_vec, choice_vec)]
  label    = 1 if correct choice, 0 if wrong choice

Saves qa_lr_w into urcm_weights.pkl so URCMSystem picks it up automatically.
Already wired in system.py:
  if self.qa_w is not None:
      z = qa_w[0]*sim + qa_w[1]*rho + qa_w[2]*mu
      a = sigmoid(z)
"""

import numpy as np
import os
import pickle
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.theory import URCMTheory

# ── 80 labelled QA examples ──────────────────────────────────────────────────
# Format: (question, correct_answer, [wrong_answers])
QA_EXAMPLES = [
    ("What do people use to absorb water?",       "paper towel",  ["spoon","plate","pen","computer"]),
    ("Where do you store dishes in a kitchen?",   "cupboard",     ["trash can","backpack","street","bed"]),
    ("What do you use to cut paper?",             "scissors",     ["spoon","plate","rope","glue"]),
    ("Where do you keep milk cold?",              "refrigerator", ["oven","desk","closet","backpack"]),
    ("What do you use to boil water?",            "kettle",       ["plate","bowl","cup","bag"]),
    ("What do you use to eat soup?",              "spoon",        ["fork","knife","plate","cup"]),
    ("What do you use to fry eggs?",              "pan",          ["bowl","kettle","spoon","cup"]),
    ("What do you use to bake bread?",            "oven",         ["microwave","fridge","kettle","pan"]),
    ("What do you sleep on?",                     "bed",          ["chair","table","floor","sofa"]),
    ("What do you sit on?",                       "chair",        ["table","bed","shelf","door"]),
    ("What do you use to watch movies?",          "television",   ["radio","phone","computer","projector"]),
    ("What do you use to call someone?",          "phone",        ["book","lamp","desk","hat"]),
    ("What do you use to tell the time?",         "clock",        ["ruler","mirror","phone","radio"]),
    ("What do you use to wake up in the morning?","alarm clock",  ["radio","lamp","fan","calendar"]),
    ("What do you use to write on paper?",        "pen",          ["pencil","marker","brush","crayon"]),
    ("What do you use to erase a mistake?",       "eraser",       ["pen","ruler","scissors","tape"]),
    ("What do you carry books to school in?",     "backpack",     ["wallet","plate","bucket","suitcase"]),
    ("What do you use to measure length?",        "ruler",        ["scale","clock","thermometer","compass"]),
    ("What do you use to brush your teeth?",      "toothbrush",   ["comb","fork","spoon","stick"]),
    ("What do you use to dry your hair?",         "hair dryer",   ["fan","towel","comb","iron"]),
    ("What protects your skin from the sun?",     "sunscreen",    ["soap","shampoo","lotion","oil"]),
    ("What do you use to check your temperature?","thermometer",  ["ruler","scale","clock","mirror"]),
    ("What do you use to sweep the floor?",       "broom",        ["mop","vacuum","rag","brush"]),
    ("What do you use to clean floors?",          "vacuum",       ["broom","rag","sponge","bucket"]),
    ("What do you use to water plants?",          "watering can", ["bucket","hose","cup","bottle"]),
    ("What do you wear in the rain?",             "raincoat",     ["jacket","sweater","scarf","hat"]),
    ("Where do fish live?",                       "water",        ["land","air","fire","ice"]),
    ("What do you need to drive a car?",          "license",      ["key","map","phone","bag"]),
    ("What do you use to ride in the sky?",       "airplane",     ["boat","train","car","bus"]),
    ("What do you put on bread?",                 "butter",       ["oil","sauce","jam","honey"]),
    ("What do you add to make food salty?",       "salt",         ["sugar","pepper","oil","vinegar"]),
    ("What do you use to make toast?",            "toaster",      ["oven","microwave","pan","grill"]),
    ("What do you drink when thirsty?",           "water",        ["juice","milk","coffee","tea"]),
    ("What do you use to draw a straight line?",  "ruler",        ["pen","pencil","compass","eraser"]),
    ("What do you use to stick paper together?",  "glue",         ["tape","stapler","pin","clip"]),
    ("Where do you keep money?",                  "wallet",       ["bag","box","drawer","shelf"]),
    ("What do you use to send an email?",         "computer",     ["phone","fax","printer","scanner"]),
    ("What do you use to dig in the garden?",     "shovel",       ["rake","hoe","spade","trowel"]),
    ("What do birds use to fly?",                 "wings",        ["fins","legs","tail","beak"]),
    ("Where do planes take off from?",            "airport",      ["station","harbour","terminal","dock"]),
    # Medical domain (from hippocampus)
    ("What is the primary risk of combining Warfarin and Aspirin?",
     "Increased bleeding risk", ["low blood pressure","rash","nausea","fever"]),
    ("How do Statins primarily work?",
     "Inhibit HMG-CoA reductase", ["block calcium","raise insulin","reduce sodium","dilate vessels"]),
    ("Which drug is a primary choice for treating acute Anaphylaxis?",
     "Epinephrine", ["aspirin","insulin","warfarin","metformin"]),
    # Legal domain (from hippocampus)
    ("What is the primary obligation of the confidentiality clause?",
     "Confidentiality", ["Payment","Assignment","Termination","Warranty"]),
    ("What does the governing law clause specify?",
     "Governing Law", ["Payment terms","Confidentiality","Assignment","Force Majeure"]),
]


def encode(pipeline, rpenc, text):
    fp = pipeline.process_text(text)
    return rpenc.encode_path(fp)


def extract_features(q_vec, c_vec):
    """3 features: cosine_sim, rho(choice), chi(q,c)"""
    sim  = float(np.dot(q_vec, c_vec) /
                 (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9))
    rho  = URCMTheory.calculate_rho(c_vec)
    chi  = URCMTheory.calculate_chi(q_vec, c_vec)
    return np.array([sim, rho, chi], dtype=np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_scorer(resonance_dim=1024, epochs=500, lr=0.1):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    wp = os.path.join(root_dir, "urcm_weights.pkl")

    pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
    rpenc    = ResonancePathEncoder(input_dim=24, resonance_dim=resonance_dim)

    if os.path.exists(wp):
        with open(wp, "rb") as f:
            wdata = pickle.load(f)
        if wdata["W_res"].shape == (resonance_dim, resonance_dim):
            rpenc.W_in  = wdata["W_in"]
            rpenc.W_res = wdata["W_res"]
            rpenc.W_out = wdata["W_out"]
            rpenc.bias  = wdata["bias"]
            print(f"Loaded weights. Hippocampus: {len(wdata.get('hippocampus',[]))}")

    print(f"\nBuilding training set from {len(QA_EXAMPLES)} QA examples...")

    X, y = [], []
    for question, correct, wrongs in QA_EXAMPLES:
        try:
            q_vec = encode(pipeline, rpenc, question)
            c_vec = encode(pipeline, rpenc, correct)
            X.append(extract_features(q_vec, c_vec))
            y.append(1.0)
            for wrong in wrongs[:2]:   # 2 negatives per positive
                w_vec = encode(pipeline, rpenc, wrong)
                X.append(extract_features(q_vec, w_vec))
                y.append(0.0)
        except Exception as e:
            print(f"  Skip: {question[:40]} — {e}")

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    print(f"Training set: {len(X)} samples ({int(y.sum())} positive, {int((1-y).sum())} negative)")

    # Normalise features
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-9
    X_norm = (X - X_mean) / X_std

    # Logistic regression — gradient descent
    w = np.zeros(3, dtype=np.float64)
    best_w, best_acc = w.copy(), 0.0

    for epoch in range(epochs):
        preds = sigmoid(X_norm @ w)
        loss  = -np.mean(y * np.log(preds + 1e-9) + (1 - y) * np.log(1 - preds + 1e-9))
        grad  = X_norm.T @ (preds - y) / len(y)
        w    -= lr * grad

        if (epoch + 1) % 100 == 0:
            acc = float(np.mean((preds >= 0.5) == y))
            print(f"  Epoch {epoch+1:4d}  loss={loss:.4f}  acc={acc:.3f}  w={w.round(4)}")
            if acc > best_acc:
                best_acc = acc
                best_w   = w.copy()

    # Denormalise weights so they work on raw (unnormalised) features
    w_raw = best_w / X_std
    print(f"\nBest accuracy: {best_acc:.3f}")
    print(f"Raw weights: sim={w_raw[0]:.4f}  rho={w_raw[1]:.4f}  chi={w_raw[2]:.4f}")

    # Save back into weights file
    with open(wp, "rb") as f:
        wdata = pickle.load(f)
    wdata["qa_lr_w"] = w_raw.astype(np.float32)
    with open(wp, "wb") as f:
        pickle.dump(wdata, f)
    print(f"Saved qa_lr_w to {wp}")
    return w_raw


def evaluate_scorer(resonance_dim=1024):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    wp = os.path.join(root_dir, "urcm_weights.pkl")
    with open(wp, "rb") as f:
        wdata = pickle.load(f)

    qa_w = wdata.get("qa_lr_w")
    if qa_w is None:
        print("No qa_lr_w found — run train first")
        return

    pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
    rpenc    = ResonancePathEncoder(input_dim=24, resonance_dim=resonance_dim)
    rpenc.W_in  = wdata["W_in"]
    rpenc.W_res = wdata["W_res"]
    rpenc.W_out = wdata["W_out"]
    rpenc.bias  = wdata["bias"]

    eval_set = [
        ("What do people use to absorb water?",
         ["spoon","paper towel","plate","pen","computer"], 1),
        ("Where do you store dishes in a kitchen?",
         ["cupboard","trash can","backpack","street","bed"], 0),
        ("What do you use to cut paper?",
         ["scissors","spoon","plate","rope","glue"], 0),
        ("What do you sleep on?",
         ["table","bed","floor","sofa","chair"], 1),
        ("What do you use to write on paper?",
         ["pen","brush","crayon","ruler","eraser"], 0),
        ("What do you use to tell the time?",
         ["radio","clock","ruler","mirror","phone"], 1),
    ]

    print("\n" + "=" * 60)
    print("SCORER EVAL (cosine + rho + chi logistic)")
    print("=" * 60)
    ok = 0
    for question, choices, answer_idx in eval_set:
        q_vec = encode(pipeline, rpenc, question)
        scores = []
        for c in choices:
            c_vec = encode(pipeline, rpenc, c)
            feats = extract_features(q_vec, c_vec)
            z     = float(qa_w @ feats)
            scores.append(sigmoid(z))
        pred    = int(np.argmax(scores))
        correct = pred == answer_idx
        ok     += int(correct)
        tag     = "PASS" if correct else "FAIL"
        print(f"  [{tag}] {question[:50]}")
        print(f"        Got: {choices[pred]!r:22s}  Want: {choices[answer_idx]!r}")

    print(f"\n  Scorer QA score: {ok}/{len(eval_set)}")
    print("=" * 60)


if __name__ == "__main__":
    train_scorer(resonance_dim=1024, epochs=500, lr=0.1)
    evaluate_scorer(resonance_dim=1024)

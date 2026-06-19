"""
Commonsense QA Hebbian Training.
Deposits 60+ everyday QA pairs into W_res so raw resonance
can answer without relying on keyword rules.

Loads existing weights (preserving medical/legal knowledge),
deposits commonsense on top, then saves back.
"""

import numpy as np
import os
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.memory import GeometricMemory
from urcm.core.safe_io import safe_load_pickle

# ── 60 commonsense QA pairs ──────────────────────────────────────────────────
# Format: (question, correct_answer, [wrong_answers])
COMMONSENSE_QA = [
    # tools / kitchen
    ("What do people use to absorb water?",       "paper towel",  ["spoon","plate","pen","computer"]),
    ("Where do you store dishes in a kitchen?",   "cupboard",     ["trash can","backpack","street","bed"]),
    ("What do you use to cut paper?",             "scissors",     ["spoon","plate","rope","glue"]),
    ("Where do you keep milk cold?",              "refrigerator", ["oven","desk","closet","backpack"]),
    ("What do you use to boil water?",            "kettle",       ["plate","bowl","cup","bag"]),
    ("What do you use to eat soup?",              "spoon",        ["fork","knife","plate","cup"]),
    ("What do you use to fry eggs?",              "pan",          ["bowl","kettle","spoon","cup"]),
    ("What do you use to bake bread?",            "oven",         ["microwave","fridge","kettle","pan"]),
    ("What do you use to wash dishes?",           "soap",         ["butter","oil","vinegar","salt"]),
    ("What do you use to dry dishes?",            "towel",        ["soap","sponge","brush","bag"]),
    ("What do you put food in to keep it fresh?", "refrigerator", ["oven","shelf","drawer","bag"]),
    ("What do you use to mix batter?",            "bowl",         ["plate","cup","tray","pot"]),
    ("What do you drink coffee from?",            "mug",          ["plate","bowl","pan","tray"]),
    ("What do you use to open a can?",            "can opener",   ["spoon","knife","fork","scissors"]),
    ("What do you use to peel vegetables?",       "peeler",       ["knife","spoon","fork","grater"]),
    # home / daily life
    ("What do you use to sweep the floor?",       "broom",        ["mop","vacuum","rag","brush"]),
    ("What do you use to clean floors?",          "vacuum",       ["broom","rag","sponge","bucket"]),
    ("What do you sleep on?",                     "bed",          ["chair","table","floor","sofa"]),
    ("What do you sit on?",                       "chair",        ["table","bed","shelf","door"]),
    ("What do you use to turn on a light?",       "switch",       ["plug","cord","bulb","socket"]),
    ("What do you use to watch movies?",          "television",   ["radio","phone","computer","projector"]),
    ("What do you use to call someone?",          "phone",        ["book","lamp","desk","hat"]),
    ("What do you use to send a letter?",         "envelope",     ["box","bag","folder","tube"]),
    ("What do you use to tell the time?",         "clock",        ["ruler","mirror","phone","radio"]),
    ("What do you use to wake up in the morning?","alarm clock",  ["radio","lamp","fan","calendar"]),
    ("What do you use to lock a door?",           "key",          ["rope","magnet","tape","bolt"]),
    ("What do you use to write on paper?",        "pen",          ["pencil","marker","brush","crayon"]),
    ("What do you use to erase a mistake?",       "eraser",       ["pen","ruler","scissors","tape"]),
    ("What do you carry books to school in?",     "backpack",     ["wallet","plate","bucket","suitcase"]),
    ("What do you use to measure length?",        "ruler",        ["scale","clock","thermometer","compass"]),
    # body / health
    ("What do you use to brush your teeth?",      "toothbrush",   ["comb","fork","spoon","stick"]),
    ("What do you use to comb your hair?",        "comb",         ["brush","toothbrush","fork","spoon"]),
    ("What do you wear on your feet?",            "shoes",        ["gloves","hat","socks","belt"]),
    ("What do you wear on your head?",            "hat",          ["belt","ring","watch","scarf"]),
    ("What do you use to dry your hair?",         "hair dryer",   ["fan","towel","comb","iron"]),
    ("What protects your skin from the sun?",     "sunscreen",    ["soap","shampoo","lotion","oil"]),
    ("What do you use to check your temperature?","thermometer",  ["ruler","scale","clock","mirror"]),
    ("What do you take when you have a headache?","painkiller",   ["vitamin","juice","water","coffee"]),
    # nature / outside
    ("What do you use to dig in the garden?",     "shovel",       ["rake","hoe","spade","trowel"]),
    ("What do you use to water plants?",          "watering can", ["bucket","hose","cup","bottle"]),
    ("What do you wear in the rain?",             "raincoat",     ["jacket","sweater","scarf","hat"]),
    ("What protects you from the sun?",           "umbrella",     ["hat","scarf","gloves","coat"]),
    ("What do you use to see far away?",          "binoculars",   ["glasses","camera","telescope","lens"]),
    ("Where do fish live?",                       "water",        ["land","air","fire","ice"]),
    ("What do birds use to fly?",                 "wings",        ["fins","legs","tail","beak"]),
    # travel / transport
    ("What do you need to drive a car?",          "license",      ["key","map","phone","bag"]),
    ("What do you buy to ride a bus?",            "ticket",       ["token","card","coin","note"]),
    ("What do you use to ride in the sky?",       "airplane",     ["boat","train","car","bus"]),
    ("What do you use to cross the ocean?",       "ship",         ["plane","car","train","bus"]),
    ("Where do planes take off from?",            "airport",      ["station","harbour","terminal","dock"]),
    # food / cooking
    ("What do you put on bread?",                 "butter",       ["oil","sauce","jam","honey"]),
    ("What do you add to make food salty?",       "salt",         ["sugar","pepper","oil","vinegar"]),
    ("What do you add to make food sweet?",       "sugar",        ["salt","pepper","oil","vinegar"]),
    ("What do you use to make coffee?",           "coffee maker", ["kettle","microwave","oven","pan"]),
    ("What do you use to make toast?",            "toaster",      ["oven","microwave","pan","grill"]),
    ("What do you drink when thirsty?",           "water",        ["juice","milk","coffee","tea"]),
    # school / work
    ("What do you use to draw a straight line?",  "ruler",        ["pen","pencil","compass","eraser"]),
    ("What do you use to stick paper together?",  "glue",         ["tape","stapler","pin","clip"]),
    ("What do you use to cut cloth?",             "scissors",     ["knife","blade","razor","saw"]),
    ("Where do you keep money?",                  "wallet",       ["bag","box","drawer","shelf"]),
    ("What do you use to pay for things?",        "money",        ["key","card","token","stamp"]),
    ("What do you use to send an email?",         "computer",     ["phone","fax","printer","scanner"]),
]

CYCLES = 50    # Reduced — 1024-dim matrices are expensive; 50 cycles is sufficient per pair


def encode(pipeline, rpenc, text):
    fp = pipeline.process_text(text)
    return rpenc.encode_path(fp)


def train_commonsense(resonance_dim=1024, cycles=CYCLES):
    print(f"Loading existing weights (resonance_dim={resonance_dim})...")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(root_dir, "urcm_weights.pkl")

    pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
    rpenc    = ResonancePathEncoder(input_dim=24, resonance_dim=resonance_dim)
    mem      = GeometricMemory(resonance_dim=resonance_dim)

    # Start from existing weights if they exist
    hippocampus = []
    if os.path.exists(weights_path):
        wdata = safe_load_pickle(weights_path)
        if wdata["W_res"].shape == (resonance_dim, resonance_dim):
            rpenc.W_in   = wdata["W_in"]
            rpenc.W_res  = wdata["W_res"]
            rpenc.W_out  = wdata["W_out"]
            rpenc.bias   = wdata["bias"]
            hippocampus  = wdata.get("hippocampus", [])
            print(f"  Loaded existing weights. Hippocampus entries: {len(hippocampus)}")
        else:
            print("  Dimension mismatch — starting fresh.")
    else:
        print("  No existing weights — starting fresh.")

    W = rpenc.W_res
    total = len(COMMONSENSE_QA)

    print(f"\nTraining on {total} QA pairs x {cycles} cycles each...")
    print("-" * 60)

    for i, (question, correct, wrongs) in enumerate(COMMONSENSE_QA):
        # Encode question and correct answer
        u_q       = encode(pipeline, rpenc, question)
        u_correct = encode(pipeline, rpenc, correct)
        u_qa      = encode(pipeline, rpenc, f"{question} {correct}")

        # Deposit: question → correct answer
        for _ in range(cycles):
            W = mem.deposit_attractor(W, u_q, u_correct)

        W = mem.shock_deposit(W, u_correct, u_q, cycles=cycles // 2)
        W = mem.shock_deposit(W, u_qa, u_correct, cycles=cycles // 2)

        t_wrong = -u_correct
        for wrong in wrongs:
            u_wrong = encode(pipeline, rpenc, f"{question} {wrong}")
            W = mem.shock_deposit(W, u_wrong, t_wrong, cycles=cycles // 3)

        # Add to hippocampus (explicit fast memory)
        hippocampus.append((u_correct, correct, {"type": "commonsense_answer", "text": correct}))
        hippocampus.append((u_q,       correct, {"type": "commonsense_question", "text": question}))

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  [{i+1:3d}/{total}] Done: {question[:50]}")

    rpenc.W_res = W
    # Recompute inverse
    try:
        rpenc.W_res_inv = np.linalg.inv(W.astype(np.float64)).astype(np.float32)
    except Exception:
        rpenc.W_res_inv = np.linalg.pinv(W.astype(np.float64)).astype(np.float32)

    # Save
    weights = {
        "W_in":       rpenc.W_in,
        "W_res":      rpenc.W_res,
        "W_out":      rpenc.W_out,
        "bias":       rpenc.bias,
        "W_res_inv":  rpenc.W_res_inv,
        "hippocampus": hippocampus,
    }
    with open(weights_path, "wb") as f:
        pickle.dump(weights, f)

    print(f"\nSaved to {weights_path}")
    print(f"Total hippocampus entries: {len(hippocampus)}")
    return weights_path


def evaluate(resonance_dim=1024):
    """Raw cosine similarity eval — no keyword rules."""
    import numpy as np
    root_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(root_dir, "urcm_weights.pkl")

    pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
    rpenc    = ResonancePathEncoder(input_dim=24, resonance_dim=resonance_dim)

    eval_set = [
        ("What do people use to absorb water?",     ["spoon","paper towel","plate","pen","computer"],   1),
        ("Where do you store dishes in a kitchen?", ["cupboard","trash can","backpack","street","bed"], 0),
        ("What do you use to cut paper?",           ["scissors","spoon","plate","rope","glue"],         0),
        # extra held-out questions not in training
        ("What do you sleep on?",                   ["table","bed","floor","sofa","chair"],             1),
        ("What do you use to write on paper?",      ["pen","brush","crayon","ruler","eraser"],          0),
        ("What do you use to tell the time?",       ["radio","clock","ruler","mirror","phone"],         1),
    ]

    print("\n" + "=" * 60)
    print("RAW RESONANCE EVAL (no keyword rules)")
    print("=" * 60)
    ok = 0
    for question, choices, answer_idx in eval_set:
        q_vec = rpenc.encode_path(pipeline.process_text(question))
        scores = []
        for c in choices:
            c_vec = rpenc.encode_path(pipeline.process_text(c))
            sim = float(np.dot(q_vec, c_vec) / (
                np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9
            ))
            scores.append(sim)
        pred = int(np.argmax(scores))
        correct = pred == answer_idx
        ok += int(correct)
        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] Q: {question[:48]}")
        print(f"         Predicted: {choices[pred]!r:20s} Expected: {choices[answer_idx]!r}")

    print(f"\n  Score: {ok}/{len(eval_set)}")
    if ok >= 4:
        print("  RESULT: Resonance has learned commonsense signal.")
    elif ok >= 2:
        print("  RESULT: Partial learning — more training pairs needed.")
    else:
        print("  RESULT: Still near-random — increase CYCLES or add more pairs.")
    print("=" * 60)


if __name__ == "__main__":
    train_commonsense(resonance_dim=1024, cycles=CYCLES)
    evaluate(resonance_dim=1024)

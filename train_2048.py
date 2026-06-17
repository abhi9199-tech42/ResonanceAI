"""
URCM 2048-Dimensional Weight Training Script.

Creates fresh weights with resonance_dim=2048 and trains:
1. Base weights: W_in, W_res (orthogonal), W_out (pseudoinverse), bias
2. Hebbian commonsense deposits into W_res
3. Logistic QA scorer (qa_lr_w)

All operations are O(D^2) per pair via shock deposit (no inner loops).
Output: urcm_weights.pkl
"""

import numpy as np
import os
import pickle
import time
from typing import List, Tuple

from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.memory import GeometricMemory
from urcm.core.theory import URCMTheory

# ── Config ────────────────────────────────────────────────────────────────────
RESONANCE_DIM = 2048
FREQ_DIM = 24
DTYPE = np.float32

# ── Commonsense QA Pairs ──────────────────────────────────────────────────────
COMMONSENSE_QA: List[Tuple] = [
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
    ("What do you use to brush your teeth?",      "toothbrush",   ["comb","fork","spoon","stick"]),
    ("What do you use to comb your hair?",        "comb",         ["brush","toothbrush","fork","spoon"]),
    ("What do you wear on your feet?",            "shoes",        ["gloves","hat","socks","belt"]),
    ("What do you wear on your head?",            "hat",          ["belt","ring","watch","scarf"]),
    ("What do you use to dry your hair?",         "hair dryer",   ["fan","towel","comb","iron"]),
    ("What protects your skin from the sun?",     "sunscreen",    ["soap","shampoo","lotion","oil"]),
    ("What do you use to check your temperature?","thermometer",  ["ruler","scale","clock","mirror"]),
    ("What do you take when you have a headache?","painkiller",   ["vitamin","juice","water","coffee"]),
    ("What do you use to dig in the garden?",     "shovel",       ["rake","hoe","spade","trowel"]),
    ("What do you use to water plants?",          "watering can", ["bucket","hose","cup","bottle"]),
    ("What do you wear in the rain?",             "raincoat",     ["jacket","sweater","scarf","hat"]),
    ("What protects you from the sun?",           "umbrella",     ["hat","scarf","gloves","coat"]),
    ("What do you use to see far away?",          "binoculars",   ["glasses","camera","telescope","lens"]),
    ("Where do fish live?",                       "water",        ["land","air","fire","ice"]),
    ("What do birds use to fly?",                 "wings",        ["fins","legs","tail","beak"]),
    ("What do you need to drive a car?",          "license",      ["key","map","phone","bag"]),
    ("What do you buy to ride a bus?",            "ticket",       ["token","card","coin","note"]),
    ("What do you use to ride in the sky?",       "airplane",     ["boat","train","car","bus"]),
    ("What do you use to cross the ocean?",       "ship",         ["plane","car","train","bus"]),
    ("Where do planes take off from?",            "airport",      ["station","harbour","terminal","dock"]),
    ("What do you put on bread?",                 "butter",       ["oil","sauce","jam","honey"]),
    ("What do you add to make food salty?",       "salt",         ["sugar","pepper","oil","vinegar"]),
    ("What do you add to make food sweet?",       "sugar",        ["salt","pepper","oil","vinegar"]),
    ("What do you use to make coffee?",           "coffee maker", ["kettle","microwave","oven","pan"]),
    ("What do you use to make toast?",            "toaster",      ["oven","microwave","pan","grill"]),
    ("What do you drink when thirsty?",           "water",        ["juice","milk","coffee","tea"]),
    ("What do you use to draw a straight line?",  "ruler",        ["pen","pencil","compass","eraser"]),
    ("What do you use to stick paper together?",  "glue",         ["tape","stapler","pin","clip"]),
    ("What do you use to cut cloth?",             "scissors",     ["knife","blade","razor","saw"]),
    ("Where do you keep money?",                  "wallet",       ["bag","box","drawer","shelf"]),
    ("What do you use to pay for things?",        "money",        ["key","card","token","stamp"]),
    ("What do you use to send an email?",         "computer",     ["phone","fax","printer","scanner"]),
]

# ── Word restructuring for multi-word answers ─────────────────────────────────
RESTRUCTURE = {
    "paper towel":  ["paper towel", "towel", "paper", "absorbent towel"],
    "alarm clock":  ["alarm clock", "alarm", "clock", "wake up device"],
    "refrigerator": ["refrigerator", "fridge", "cold storage", "food cooler"],
    "can opener":   ["can opener", "opener", "tin opener"],
    "hair dryer":   ["hair dryer", "dryer", "blow dryer"],
    "watering can": ["watering can", "watering", "garden can"],
    "toothbrush":   ["toothbrush", "brush", "teeth brush"],
    "television":   ["television", "tv", "screen"],
    "coffee maker": ["coffee maker", "coffee machine"],
}

DEPOSIT_CYCLES = 800
REPEL_CYCLES = 300


def _restructure(phrase: str) -> List[str]:
    if phrase in RESTRUCTURE:
        return RESTRUCTURE[phrase]
    words = phrase.split()
    if len(words) == 1:
        return [phrase]
    return [phrase] + words


def encode(pipeline, rpenc, text: str) -> np.ndarray:
    fp = pipeline.process_text(text)
    return rpenc.encode_path(fp).astype(np.float64)


def spectral_clip(W: np.ndarray, max_sr: float = 1.2) -> np.ndarray:
    v = np.random.randn(W.shape[0]).astype(np.float64)
    v /= np.linalg.norm(v)
    for _ in range(10):
        v = W.astype(np.float64) @ v
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            break
        v /= nrm
    sr = float(np.linalg.norm(W.astype(np.float64) @ v))
    if sr > max_sr:
        W = (W.astype(np.float64) * (max_sr / sr)).astype(np.float32)
    return W


def init_base_weights():
    """Initialize fresh base weights with resonance_dim=2048."""
    print(f"Initializing base weights (D={RESONANCE_DIM})...")
    np.random.seed(42)

    # W_in: random projection (input_dim -> resonance_dim), scaled by 0.1
    W_in = np.random.normal(0, 0.1, (FREQ_DIM, RESONANCE_DIM)).astype(DTYPE)

    # W_res: orthogonal matrix scaled by 0.95 (fading memory)
    H = np.random.randn(RESONANCE_DIM, RESONANCE_DIM)
    Q, R = np.linalg.qr(H)
    W_res = (Q * 0.95).astype(DTYPE)

    # W_out: pseudoinverse of W_in (decoder)
    W_out = np.linalg.pinv(W_in.astype(np.float64)).astype(DTYPE)

    # bias: small random
    bias = np.random.normal(0, 0.01, RESONANCE_DIM).astype(DTYPE)

    # W_res_inv: exact inverse of W_res
    try:
        W_res_inv = np.linalg.inv(W_res.astype(np.float64)).astype(DTYPE)
    except np.linalg.LinAlgError:
        W_res_inv = np.linalg.pinv(W_res.astype(np.float64)).astype(DTYPE)

    print(f"  W_in:   {W_in.shape}")
    print(f"  W_res:  {W_res.shape}")
    print(f"  W_out:  {W_out.shape}")
    print(f"  bias:   {bias.shape}")

    return W_in, W_res, W_out, bias, W_res_inv


def train_hebbian(pipeline, rpenc, mem, W_res, hippocampus):
    """Hebbian shock deposits for commonsense QA pairs."""
    print(f"\nHebbian training: {len(COMMONSENSE_QA)} pairs, {DEPOSIT_CYCLES} cycles...")
    t0 = time.time()

    for i, (question, correct, wrongs) in enumerate(COMMONSENSE_QA):
        q_vec = encode(pipeline, rpenc, question)

        # Encode answer under all restructured phrasings
        phrasings = _restructure(correct)
        c_vecs = [encode(pipeline, rpenc, p) for p in phrasings]
        c_vec = np.mean(np.stack(c_vecs), axis=0)
        c_vec /= (np.linalg.norm(c_vec) + 1e-9)

        u_qa = encode(pipeline, rpenc, f"{question} {correct}")

        # 1. question -> correct
        W_res = mem.shock_deposit(W_res, q_vec, c_vec, cycles=DEPOSIT_CYCLES)

        # 2. repel wrong answers
        for wrong in wrongs:
            w_vec = encode(pipeline, rpenc, wrong)
            W_res = mem.shock_deposit(W_res, w_vec, -q_vec, cycles=REPEL_CYCLES)

        # 3. correct -> question (bidirectional)
        W_res = mem.shock_deposit(W_res, c_vec, q_vec, cycles=DEPOSIT_CYCLES // 2)

        # 4. combined phrase -> correct
        W_res = mem.shock_deposit(W_res, u_qa, c_vec, cycles=DEPOSIT_CYCLES // 2)

        # 5. restructured phrasings -> correct
        for pv in c_vecs[1:]:
            W_res = mem.shock_deposit(W_res, pv, c_vec, cycles=DEPOSIT_CYCLES // 4)

        # hippocampus entries
        hippocampus.append((c_vec.astype(DTYPE), correct, {"type": "commonsense", "text": correct}))
        hippocampus.append((q_vec.astype(DTYPE), correct, {"type": "question", "text": question}))

        if (i + 1) % 10 == 0 or (i + 1) == len(COMMONSENSE_QA):
            elapsed = time.time() - t0
            print(f"  [{i+1:3d}/{len(COMMONSENSE_QA)}] {question[:50]}  ({elapsed:.1f}s)")

    # Restore rank after Hebbian deposits (prevents attractor collapse)
    # Hebbian rank-1 updates collapse W_res to low rank, causing all long inputs
    # to converge to the same attractor. Adding small noise preserves full-rank.
    print("Restoring rank...")
    np.random.seed(42)
    noise_std = float(np.std(W_res)) * 0.05
    noise = np.random.normal(0, noise_std, W_res.shape).astype(DTYPE)
    W_res += noise

    # Spectral clip to stable regime (prevents chaotic blowup)
    print("Spectral clipping...")
    W_res = spectral_clip(W_res, max_sr=0.95)

    return W_res, hippocampus


def train_qa_scorer(pipeline, rpenc, epochs=800, lr=0.05):
    """Train 5-feature logistic regression for QA scoring."""
    print("\nTraining QA scorer (5 features)...")

    def _features(q_vec, c_vec):
        q64 = q_vec.astype(np.float64)
        c64 = c_vec.astype(np.float64)
        sim = float(np.dot(q64, c64) / (np.linalg.norm(q64) * np.linalg.norm(c64) + 1e-9))
        rho = URCMTheory.calculate_rho(c64)
        chi = URCMTheory.calculate_chi(q64, c64)
        rho_q = URCMTheory.calculate_rho(q64)
        norm_c = float(np.linalg.norm(c64))
        return np.array([sim, rho, chi, rho_q, norm_c], dtype=np.float64)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    X, y = [], []
    for question, correct, wrongs in COMMONSENSE_QA:
        try:
            q_vec = encode(pipeline, rpenc, question)
            c_vec = encode(pipeline, rpenc, correct)
            X.append(_features(q_vec, c_vec))
            y.append(1.0)
            for wrong in wrongs[:3]:
                w_vec = encode(pipeline, rpenc, wrong)
                X.append(_features(q_vec, w_vec))
                y.append(0.0)
        except Exception:
            pass

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    mu_X = X.mean(axis=0)
    sd_X = X.std(axis=0) + 1e-9
    X_n = (X - mu_X) / sd_X

    w = np.zeros(X.shape[1], dtype=np.float64)
    best_w, best_acc = w.copy(), 0.0

    for ep in range(epochs):
        p = sigmoid(X_n @ w)
        grad = X_n.T @ (p - y) / len(y)
        w -= lr * grad
        acc = float(np.mean((p >= 0.5) == y))
        if acc > best_acc:
            best_acc = acc
            best_w = w.copy()

    w_raw = (best_w / sd_X).astype(DTYPE)
    print(f"  Scorer accuracy: {best_acc:.3f}  weights: {w_raw.round(4)}")
    return w_raw


def evaluate(pipeline, rpenc, qa_lr_w):
    """Evaluate using the system's QA dynamics (not raw vector comparison)."""
    from urcm.core.system import URCMSystem

    # Build a lightweight system for evaluation
    system = URCMSystem(resonance_dim=RESONANCE_DIM, use_wave_dynamics=False)
    system.encoder.W_in = rpenc.W_in
    system.encoder.W_res = rpenc.W_res
    system.encoder.W_out = rpenc.W_out
    system.encoder.bias = rpenc.bias

    eval_set = [
        ("What do people use to absorb water?",
         ["spoon", "paper towel", "plate", "pen", "computer"], 1),
        ("Where do you store dishes in a kitchen?",
         ["cupboard", "trash can", "backpack", "street", "bed"], 0),
        ("What do you use to cut paper?",
         ["scissors", "spoon", "plate", "rope", "glue"], 0),
        ("What do you sleep on?",
         ["table", "bed", "floor", "sofa", "chair"], 1),
        ("What do you use to write on paper?",
         ["pen", "brush", "crayon", "ruler", "eraser"], 0),
        ("What do you use to tell the time?",
         ["radio", "clock", "ruler", "mirror", "phone"], 1),
        ("What do you use to boil water?",
         ["plate", "bowl", "kettle", "cup", "bag"], 2),
        ("What do you drink when thirsty?",
         ["juice", "milk", "water", "coffee", "tea"], 2),
    ]

    print("\n" + "=" * 65)
    print("EVALUATION (using system QA dynamics)")
    print("=" * 65)
    ok_system = 0
    for question, choices, answer_idx in eval_set:
        result = system.solve_qa_right_brain(question, choices)
        pred = choices.index(result["winner"]) if result["winner"] in choices else -1
        ok_system += int(pred == answer_idx)
        tag = "PASS" if pred == answer_idx else "FAIL"
        print(f"  [{tag}] {question[:50]}")
        print(f"    Got: {result['winner']!r:22s}  Want: {choices[answer_idx]!r}")

    print(f"\n  System QA score: {ok_system}/{len(eval_set)}")
    print("=" * 65)


def main():
    t_start = time.time()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    wp = os.path.join(root_dir, "urcm_weights.pkl")

    pipeline = PhonemeFrequencyPipeline(frequency_dim=FREQ_DIM)
    rpenc = ResonancePathEncoder(
        input_dim=FREQ_DIM,
        resonance_dim=RESONANCE_DIM,
        use_wave_dynamics=True
    )
    mem = GeometricMemory(resonance_dim=RESONANCE_DIM)

    # 1. Initialize base weights
    W_in, W_res, W_out, bias, W_res_inv = init_base_weights()
    rpenc.W_in = W_in
    rpenc.W_res = W_res
    rpenc.W_out = W_out
    rpenc.bias = bias
    rpenc.W_res_inv = W_res_inv

    # 2. Hebbian training
    hippocampus = []
    W_res, hippocampus = train_hebbian(pipeline, rpenc, mem, W_res, hippocampus)
    rpenc.W_res = W_res

    # 2b. Re-encode all hippocampus entries with final W_res
    # Without this, stored vectors were encoded with INITIAL W_res (before deposits)
    # but the saved W_res is the final (post-deposit) version — mismatch causes
    # re-encoded queries to give cosine ~0.12 vs stored vectors.
    print(f"Re-encoding {len(hippocampus)} hippocampus entries with final W_res...")
    new_hippocampus = []
    for mem_vec, label, meta in hippocampus:
        text = meta.get("text", label)
        vec = encode(pipeline, rpenc, text)
        new_hippocampus.append((vec, label, meta))
    hippocampus = new_hippocampus
    print(f"  Done — {len(hippocampus)} entries re-encoded.")

    # Recompute inverse
    print("Computing W_res_inv...")
    try:
        rpenc.W_res_inv = np.linalg.inv(W_res.astype(np.float64)).astype(DTYPE)
    except np.linalg.LinAlgError:
        rpenc.W_res_inv = np.linalg.pinv(W_res.astype(np.float64)).astype(DTYPE)

    # 3. Train QA scorer
    qa_lr_w = train_qa_scorer(pipeline, rpenc)

    # 4. Save
    weights = {
        "W_in": rpenc.W_in,
        "W_res": rpenc.W_res,
        "W_out": rpenc.W_out,
        "bias": rpenc.bias,
        "W_res_inv": rpenc.W_res_inv,
        "hippocampus": hippocampus,
        "qa_lr_w": qa_lr_w,
    }
    with open(wp, "wb") as f:
        pickle.dump(weights, f)

    total = time.time() - t_start
    print(f"\nSaved to {wp}")
    print(f"Hippocampus: {len(hippocampus)} entries")
    print(f"Total time: {total:.1f}s")

    # 5. Evaluate
    evaluate(pipeline, rpenc, qa_lr_w)


if __name__ == "__main__":
    main()

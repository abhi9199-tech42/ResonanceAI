"""
Commonsense Hebbian Shock Training — stable float64 + word restructuring + logistic scorer.

Three improvements:
1. float64 throughout + spectral clip after all deposits (stable)
2. Word restructuring — each answer deposited under multiple phrasings
   so multi-word answers like "paper towel" get full coverage
3. Logistic scorer (qa_lr_w) trained at the end and saved into weights
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

# ── Training pairs ────────────────────────────────────────────────────────────
# Format: (question, correct_answer, [wrong_answers])
BOOST_PAIRS: List[Tuple] = [
    ("What do people use to absorb water?",       "paper towel",  ["spoon","plate","pen","computer"]),
    ("Where do you store dishes in a kitchen?",   "cupboard",     ["trash can","backpack","street","bed"]),
    ("What do you sleep on?",                     "bed",          ["chair","table","floor","sofa"]),
    ("What do you use to write on paper?",        "pen",          ["brush","crayon","ruler","eraser"]),
    ("What do you use to tell the time?",         "clock",        ["radio","ruler","mirror","phone"]),
    ("What do you use to cut paper?",             "scissors",     ["spoon","plate","rope","glue"]),
    ("Where do you keep milk cold?",              "refrigerator", ["oven","desk","closet","backpack"]),
    ("What do you use to boil water?",            "kettle",       ["plate","bowl","cup","bag"]),
    ("What do you use to eat soup?",              "spoon",        ["fork","knife","plate","cup"]),
    ("What do you use to fry eggs?",              "pan",          ["bowl","kettle","spoon","cup"]),
    ("What do you use to watch movies?",          "television",   ["radio","phone","computer","projector"]),
    ("What do you use to call someone?",          "phone",        ["book","lamp","desk","hat"]),
    ("What do you use to wake up in the morning?","alarm clock",  ["radio","lamp","fan","calendar"]),
    ("What do you use to erase a mistake?",       "eraser",       ["pen","ruler","scissors","tape"]),
    ("What do you carry books to school in?",     "backpack",     ["wallet","plate","bucket","suitcase"]),
    ("What do you use to brush your teeth?",      "toothbrush",   ["comb","fork","spoon","stick"]),
    ("What do you use to dry your hair?",         "hair dryer",   ["fan","towel","comb","iron"]),
    ("What do you use to sweep the floor?",       "broom",        ["mop","vacuum","rag","brush"]),
    ("What do you use to clean floors?",          "vacuum",       ["broom","rag","sponge","bucket"]),
    ("What do you drink when thirsty?",           "water",        ["juice","milk","coffee","tea"]),
    ("What do you add to make food salty?",       "salt",         ["sugar","pepper","oil","vinegar"]),
    ("What do you use to make toast?",            "toaster",      ["oven","microwave","pan","grill"]),
    ("Where do you keep money?",                  "wallet",       ["bag","box","drawer","shelf"]),
    ("What absorbs liquid spills?",               "paper towel",  ["spoon","plate","pen","cup"]),
    ("Where are dishes stored at home?",          "cupboard",     ["trash","floor","bed","bag"]),
    ("What do you rest on at night?",             "bed",          ["sofa","chair","table","floor"]),
    ("What instrument shows the hour?",           "clock",        ["ruler","scale","mirror","phone"]),
    ("What tool writes ink on paper?",            "pen",          ["pencil","eraser","ruler","brush"]),
    ("What cuts paper and cloth?",                "scissors",     ["knife","blade","razor","saw"]),
    ("What keeps food cold?",                     "refrigerator", ["oven","microwave","shelf","drawer"]),
    ("What do you use to open a can?",            "can opener",   ["spoon","knife","fork","scissors"]),
    ("What do you use to measure weight?",        "scale",        ["ruler","clock","thermometer","compass"]),
    ("What do you wear on your feet?",            "shoes",        ["gloves","hat","socks","belt"]),
    ("What do you use to water plants?",          "watering can", ["bucket","hose","cup","bottle"]),
    ("What do you use to wash dishes?",           "soap",         ["butter","oil","vinegar","salt"]),
    ("What do you use to dry dishes?",            "towel",        ["soap","sponge","brush","bag"]),
    ("What do you drink coffee from?",            "mug",          ["plate","bowl","pan","tray"]),
    ("What do you use to mix ingredients?",       "bowl",         ["plate","cup","tray","pot"]),
    ("What do you use to cut bread?",             "knife",        ["scissors","spoon","fork","spatula"]),
    ("What do you use to lock a door?",           "key",          ["rope","magnet","tape","bolt"]),
]

CYCLES       = 800
REPEL_CYCLES = 300


def _restructure(phrase: str) -> List[str]:
    """
    Return multiple encodable phrasings of a word or phrase.
    Helps multi-word answers get deposited from multiple angles.

    Examples:
      "paper towel" -> ["paper towel", "towel", "paper", "absorbent towel"]
      "alarm clock" -> ["alarm clock", "alarm", "clock"]
      "refrigerator" -> ["refrigerator", "fridge", "cold storage"]
    """
    variants = {
        "paper towel":  ["paper towel", "towel", "paper", "absorbent towel"],
        "alarm clock":  ["alarm clock", "alarm", "clock", "wake up device"],
        "refrigerator": ["refrigerator", "fridge", "cold storage", "food cooler"],
        "can opener":   ["can opener", "opener", "tin opener"],
        "hair dryer":   ["hair dryer", "dryer", "blow dryer"],
        "watering can": ["watering can", "watering", "garden can"],
        "toothbrush":   ["toothbrush", "brush", "teeth brush"],
        "television":   ["television", "tv", "screen"],
        "alarm":        ["alarm", "alert", "signal"],
        "scissors":     ["scissors", "shears", "cutter"],
        "backpack":     ["backpack", "bag", "school bag"],
        "eraser":       ["eraser", "rubber", "correction tool"],
        "vacuum":       ["vacuum", "vacuum cleaner", "hoover"],
        "toaster":      ["toaster", "toast maker", "bread toaster"],
    }
    if phrase in variants:
        return variants[phrase]
    # For single words or unknown phrases: use as-is plus individual words
    words = phrase.split()
    if len(words) == 1:
        return [phrase]
    return [phrase] + words   # e.g. "trash can" -> ["trash can", "trash", "can"]


def encode(pipeline, rpenc, text: str) -> np.ndarray:
    fp = pipeline.process_text(text)
    return rpenc.encode_path(fp).astype(np.float64)


def shock_deposit_stable(W: np.ndarray, u: np.ndarray, v: np.ndarray,
                          cycles: int, mem: GeometricMemory) -> np.ndarray:
    """
    Stable float64 shock deposit.
    Computes the rank-1 update in float64, clips W to [-5, 5], returns float32.
    """
    u64 = u.astype(np.float64)
    v64 = v.astype(np.float64)
    W64 = W.astype(np.float64)

    norm_u_sq = float(np.dot(u64, u64))
    if norm_u_sq < 1e-8:
        return W

    safe_v = np.clip(v64, -0.95, 0.95)
    target = np.arctanh(safe_v)
    error  = target - np.dot(u64, W64)

    scale  = float(cycles) / norm_u_sq
    update = np.outer(u64, error) * scale
    W_new  = np.clip(W64 + update, -5.0, 5.0).astype(np.float32)

    mem.deposited_count += cycles
    return W_new


def spectral_clip(W: np.ndarray, max_sr: float = 1.2) -> np.ndarray:
    """Clip spectral radius of W to max_sr without full eigendecomposition."""
    # Power iteration estimate (fast, avoids O(D^3) full SVD)
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


# ── Logistic scorer ───────────────────────────────────────────────────────────

def _features(q_vec, c_vec):
    """5 features for logistic scorer."""
    q64 = q_vec.astype(np.float64)
    c64 = c_vec.astype(np.float64)
    sim  = float(np.dot(q64, c64) /
                 (np.linalg.norm(q64) * np.linalg.norm(c64) + 1e-9))
    rho  = URCMTheory.calculate_rho(c64)
    chi  = URCMTheory.calculate_chi(q64, c64)
    rho_q = URCMTheory.calculate_rho(q64)
    norm_c = float(np.linalg.norm(c64))
    return np.array([sim, rho, chi, rho_q, norm_c], dtype=np.float64)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_logistic_scorer(pipeline, rpenc, epochs=800, lr=0.05):
    """Train 5-weight logistic regression on the boost pairs."""
    X, y = [], []
    for question, correct, wrongs in BOOST_PAIRS:
        try:
            q_vec = encode(pipeline, rpenc, question)
            c_vec = encode(pipeline, rpenc, correct)
            X.append(_features(q_vec, c_vec));  y.append(1.0)
            for wrong in wrongs[:3]:
                w_vec = encode(pipeline, rpenc, wrong)
                X.append(_features(q_vec, w_vec)); y.append(0.0)
        except Exception:
            pass

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # Normalise
    mu_X  = X.mean(axis=0)
    sd_X  = X.std(axis=0) + 1e-9
    X_n   = (X - mu_X) / sd_X

    w = np.zeros(X.shape[1], dtype=np.float64)
    best_w, best_acc = w.copy(), 0.0

    for ep in range(epochs):
        p    = _sigmoid(X_n @ w)
        grad = X_n.T @ (p - y) / len(y)
        w   -= lr * grad
        acc  = float(np.mean((p >= 0.5) == y))
        if acc > best_acc:
            best_acc = acc
            best_w   = w.copy()

    # Denormalise so scorer works on raw features
    w_raw = (best_w / sd_X).astype(np.float32)
    print(f"  Scorer accuracy: {best_acc:.3f}  weights: {w_raw.round(4)}")
    return w_raw


# ── Main ──────────────────────────────────────────────────────────────────────

def boost_and_score():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    wp = os.path.join(root_dir, "urcm_weights.pkl")

    pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
    rpenc    = ResonancePathEncoder(input_dim=24, resonance_dim=1024)
    mem      = GeometricMemory(resonance_dim=1024)

    with open(wp, "rb") as f:
        wdata = pickle.load(f)
    rpenc.W_in  = wdata["W_in"]
    rpenc.W_res = wdata["W_res"]
    rpenc.W_out = wdata["W_out"]
    rpenc.bias  = wdata["bias"]
    hippocampus = list(wdata.get("hippocampus", []))
    print(f"Loaded. Hippocampus: {len(hippocampus)} entries")
    print(f"Shock-training {len(BOOST_PAIRS)} pairs "
          f"(cycles={CYCLES}, repel={REPEL_CYCLES}, stable float64)\n")

    W  = rpenc.W_res
    t0 = time.time()

    for i, (question, correct, wrongs) in enumerate(BOOST_PAIRS):
        q_vec  = encode(pipeline, rpenc, question)
        # Encode answer under all restructured phrasings
        ans_phrasings = _restructure(correct)
        c_vecs = [encode(pipeline, rpenc, p) for p in ans_phrasings]
        # Mean of phrasings = stable composite vector
        c_vec  = np.mean(np.stack(c_vecs), axis=0)
        c_vec /= (np.linalg.norm(c_vec) + 1e-9)

        u_qa   = encode(pipeline, rpenc, f"{question} {correct}")

        # Wrong answer vectors
        w_vecs = [encode(pipeline, rpenc, w) for w in wrongs]

        # 1. question → correct (with repulsion)
        W = shock_deposit_stable(W, q_vec, c_vec, CYCLES, mem)
        for wv in w_vecs:
            W = shock_deposit_stable(W, wv, -q_vec, REPEL_CYCLES, mem)

        # 2. correct → question (bidirectional)
        W = shock_deposit_stable(W, c_vec, q_vec, CYCLES // 2, mem)

        # 3. combined phrase → correct
        W = shock_deposit_stable(W, u_qa, c_vec, CYCLES // 2, mem)

        # 4. each restructured phrasing → correct
        for phrasing, pv in zip(ans_phrasings[1:], c_vecs[1:]):
            W = shock_deposit_stable(W, pv, c_vec, CYCLES // 4, mem)

        hippocampus.append((c_vec.astype(np.float32), correct,
                            {"type": "shock_answer", "text": correct}))
        hippocampus.append((q_vec.astype(np.float32), correct,
                            {"type": "shock_question", "text": question}))

        print(f"  [{i+1:2d}/{len(BOOST_PAIRS)}] {question[:55]}"
              f"  ({time.time()-t0:.1f}s)")

    # Spectral clip — keep W_res stable
    print("\nSpectral clip...")
    W = spectral_clip(W, max_sr=1.2)
    rpenc.W_res = W

    try:
        rpenc.W_res_inv = np.linalg.inv(W.astype(np.float64)).astype(np.float32)
    except Exception:
        rpenc.W_res_inv = np.linalg.pinv(W.astype(np.float64)).astype(np.float32)

    # Train logistic scorer on updated geometry
    print("\nTraining logistic scorer (5 features)...")
    qa_lr_w = train_logistic_scorer(pipeline, rpenc)

    weights = {
        "W_in": rpenc.W_in, "W_res": rpenc.W_res,
        "W_out": rpenc.W_out, "bias": rpenc.bias,
        "W_res_inv": rpenc.W_res_inv,
        "hippocampus": hippocampus,
        "qa_lr_w": qa_lr_w,
    }
    with open(wp, "wb") as f:
        pickle.dump(weights, f)

    total = time.time() - t0
    print(f"\nSaved. Total hippocampus: {len(hippocampus)}  Time: {total:.1f}s")


def evaluate():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    wp = os.path.join(root_dir, "urcm_weights.pkl")
    with open(wp, "rb") as f:
        wdata = pickle.load(f)

    pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
    rpenc    = ResonancePathEncoder(input_dim=24, resonance_dim=1024)
    rpenc.W_in  = wdata["W_in"]
    rpenc.W_res = wdata["W_res"]
    rpenc.W_out = wdata["W_out"]
    rpenc.bias  = wdata["bias"]
    qa_lr_w     = wdata.get("qa_lr_w")
    if qa_lr_w is None:
        print("No scorer weights found. Run boost_and_score() first.")
        return

    # Use the actual number of features the scorer was trained on
    scorer_dim = len(qa_lr_w)
    print(f"  Using scorer with {scorer_dim} features")

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
        ("What do you use to boil water?",
         ["plate","bowl","kettle","cup","bag"], 2),
        ("What do you drink when thirsty?",
         ["juice","milk","water","coffee","tea"], 2),
    ]

    print("\n" + "=" * 65)
    print("EVAL — Raw resonance  |  With logistic scorer")
    print("=" * 65)

    ok_raw = ok_scored = 0
    for question, choices, answer_idx in eval_set:
        q_vec = encode(pipeline, rpenc, question)

        raw_scores    = []
        scored_scores = []
        for c in choices:
            c_vec = encode(pipeline, rpenc, c)
            sim   = float(np.dot(q_vec, c_vec) /
                          (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9))
            if qa_lr_w is not None:
                feats = _features(q_vec, c_vec)
                scorer_dim = len(qa_lr_w)
                if len(feats) > scorer_dim:
                    feats = feats[:scorer_dim]
                scored_scores.append(float(_sigmoid(float(qa_lr_w @ feats))))
            else:
                scored_scores.append(sim)


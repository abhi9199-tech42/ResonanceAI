import sys
import os
import sqlite3
import numpy as np
import pickle
from typing import List, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from urcm.core.hierarchical_encoder import HierarchicalEncoder
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.memory import GeometricMemory
from urcm.core.sanskrit_grammar import SanskritGrammar
from urcm.core.web_sensor import WebSensor
from urcm.core.resonance_encoder import ResonancePathEncoder


def text_to_concept(encoder: HierarchicalEncoder, pipeline: PhonemeFrequencyPipeline, text: str) -> np.ndarray:
    fp = pipeline.process_text(text)
    l2_vec, _ = encoder.encode_concept(fp.vectors)
    return l2_vec


def average_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def train(db_path: str, brain_path: str = "urcm_identity.pkl", l2_dim: int = 512):
    pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
    memory = GeometricMemory(resonance_dim=l2_dim)
    grammar = SanskritGrammar()
    rpenc = ResonancePathEncoder(input_dim=24, resonance_dim=l2_dim)
    W_in_det = np.zeros((24, l2_dim))
    for i in range(24):
        W_in_det[i, i] = 1.0
    rpenc.W_in = W_in_det
    rpenc.W_out = np.linalg.pinv(W_in_det)
    W = rpenc.W_res
    concept_map = {}  # optional for ReasoningEngine, not used by URCMSystem
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute("SELECT question, answer FROM qa").fetchall()
    seed_pairs = [
        ("What do people use to absorb water?", "paper towel"),
        ("Where do you store dishes in a kitchen?", "cupboard"),
        ("What do you need to drive to work?", "car"),
        ("What do you use to cut paper?", "scissors"),
        ("What do you use to write a letter?", "pen"),
        ("Where do you keep milk cold?", "refrigerator"),
        ("Where do you watch a movie at home?", "television"),
        ("What lights up a dark room?", "lamp"),
        ("What protects hands when baking?", "oven mitts"),
        ("What do you eat cereal with?", "bowl"),
        ("What do you use to clean your teeth?", "toothbrush"),
        ("What opens a can?", "can opener"),
        ("What measures time?", "clock"),
        ("What measures temperature?", "thermometer"),
        ("What do you use to call a friend?", "phone"),
        ("What carries books to school?", "backpack"),
        ("What helps you see far things?", "binoculars"),
        ("What do you use to pay for items?", "money"),
        ("What plays music?", "radio"),
        ("What helps you see at night while walking?", "flashlight"),
        ("What dries wet hair?", "hair dryer"),
        ("What do cooks read to check ingredients?", "cookbook"),
        ("What do you use to dig a hole?", "shovel"),
        ("What helps build a sandcastle?", "bucket"),
        ("What prevents sunburn?", "sunscreen"),
        ("What keeps your head warm in winter?", "hat"),
        ("What makes coffee quickly?", "coffee maker"),
        ("What serves soup?", "ladle"),
        ("What boils water on the counter?", "kettle"),
        ("What cleans floors efficiently?", "vacuum"),
        ("What helps fix a leaky pipe?", "wrench"),
        ("What wakes you up in the morning?", "alarm clock"),
        ("What removes wrinkles from clothes?", "iron"),
        ("What holds papers together?", "stapler"),
        ("What cuts wood?", "saw"),
        ("What erases pencil marks?", "eraser"),
        ("What waters plants?", "watering can"),
        ("What protects eyes from bright sun?", "sunglasses"),
        ("What cleans the floor with water?", "mop"),
        ("Which device toasts bread?", "toaster"),
        ("Which tool tightens a nut?", "wrench"),
        ("Which gadget wakes you at a set time?", "alarm clock"),
        ("Which item absorbs spills on the counter?", "paper towel"),
        ("Which device blends smoothies?", "blender"),
        ("Which tool drives screws?", "screwdriver"),
        ("Which tool checks if a shelf is straight?", "level"),
        ("Which container holds soup while eating?", "bowl"),
        ("Which appliance dries clothes?", "dryer"),
        ("Which device illuminates a dark hallway at night?", "flashlight"),
        ("Which utensil flips pancakes?", "spatula"),
        ("Which tool grates cheese?", "grater"),
        ("Which appliance keeps food cold?", "refrigerator"),
        ("Which item protects hands from heat while cooking?", "oven mitts"),
        ("Which device measures weight?", "scale"),
        ("Which utensil removes the outer skin of vegetables?", "peeler"),
        ("Which item heats water quickly for tea?", "kettle"),
    ]
    paraphrase_pairs = [
        ("Where are dishes stored in the kitchen?", "cupboard"),
        ("Where can you store dishes in the kitchen?", "cupboard"),
        ("Where do people keep dishes?", "cupboard"),
        ("What absorbs liquid spills on a counter?", "paper towel"),
        ("What absorbs water spills?", "paper towel"),
        ("Which tool is used to cut paper?", "scissors"),
        ("Which thing cuts paper?", "scissors"),
        ("Where do you keep food cold?", "refrigerator"),
        ("Where do you keep milk cold in the kitchen?", "refrigerator"),
    ]
    qa_choices = {
        "What do people use to absorb water?": ["spoon","paper towel","plate","pen","computer"],
        "Where do you store dishes in a kitchen?": ["cupboard","trash can","backpack","street","bed"],
        "What do you need to drive to work?": ["car","spoon","bicycle","paper","candle"],
        "What do you use to cut paper?": ["scissors","spoon","plate","rope","glue"],
        "What do you use to write a letter?": ["pen","knife","bowl","shoe","hammer"],
        "Where do you keep milk cold?": ["refrigerator","oven","desk","closet","backpack"],
        "Where do you watch a movie at home?": ["television","microwave","toaster","sink","vacuum"],
        "What lights up a dark room?": ["lamp","blanket","book","rock","pillow"],
        "What protects hands when baking?": ["oven mitts","scarf","gloves","hat","belt"],
        "What do you eat cereal with?": ["bowl","box","napkin","pan","envelope"],
        "What do you use to clean your teeth?": ["toothbrush","comb","rake","spoon","brush"],
        "What opens a can?": ["can opener","pencil","tape","fork","drill"],
        "What measures time?": ["clock","spoon","mirror","door","radio"],
        "What measures temperature?": ["thermometer","ruler","scale","glass","cup"],
        "What do you use to call a friend?": ["phone","book","lamp","desk","hat"],
        "What carries books to school?": ["backpack","plate","suitcase","wallet","bucket"],
        "What helps you see far things?": ["binoculars","fork","keyboard","sponge","spatula"],
        "What do you use to pay for items?": ["money","stick","paper clip","soap","string"],
        "What plays music?": ["radio","ladder","broom","plate","mop"],
        "What helps you see at night while walking?": ["flashlight","newspaper","wallet","pencil","glove"],
        "What dries wet hair?": ["hair dryer","fan","comb","brush","hat"],
        "What do cooks read to check ingredients?": ["cookbook","calendar","magazine","ticket","receipt"],
        "What do you use to dig a hole?": ["shovel","pan","book","spoon","rope"],
        "What helps build a sandcastle?": ["bucket","mirror","chair","plate","brush"],
        "What prevents sunburn?": ["sunscreen","soap","shampoo","glue","ink"],
        "What keeps your head warm in winter?": ["hat","belt","ring","watch","scarf"],
        "What makes coffee quickly?": ["coffee maker","microwave","stove","pan","cup"],
        "What serves soup?": ["ladle","knife","spoon","fork","straw"],
        "What boils water on the counter?": ["kettle","bowl","plate","bucket","cup"],
        "What cleans floors efficiently?": ["vacuum","pencil","paper","book","pan"],
        "What helps fix a leaky pipe?": ["wrench","spoon","scissors","pen","tape"],
        "What wakes you up in the morning?": ["alarm clock","mirror","toaster","broom","bucket"],
        "What removes wrinkles from clothes?": ["iron","comb","brush","razor","fan"],
        "What holds papers together?": ["stapler","spoon","plate","stick","book"],
        "What cuts wood?": ["saw","scissors","knife","pen","spoon"],
        "What erases pencil marks?": ["eraser","soap","towel","brush","comb"],
        "What waters plants?": ["watering can","bucket","plate","glass","lamp"],
        "What protects eyes from bright sun?": ["sunglasses","hat","gloves","scarf","belt"],
        "What cleans the floor with water?": ["mop","broom","vacuum","rag","brush"],
        "Which device toasts bread?": ["toaster","microwave","stove","kettle","iron"],
        "Which tool tightens a nut?": ["wrench","scissors","hammer","knife","spoon"],
        "Which gadget wakes you at a set time?": ["alarm clock","radio","vacuum","lamp","calendar"],
        "Which item absorbs spills on the counter?": ["paper towel","plate","bowl","sponge","cup"],
        "Which device blends smoothies?": ["blender","mixer","toaster","kettle","microwave"],
        "Which tool drives screws?": ["screwdriver","wrench","saw","hammer","pliers"],
        "Which tool checks if a shelf is straight?": ["level","ruler","compass","thermometer","scale"],
        "Which container holds soup while eating?": ["bowl","plate","jar","box","bag"],
        "Which appliance dries clothes?": ["dryer","washer","vacuum","iron","fan"],
        "Which device illuminates a dark hallway at night?": ["flashlight","lamp","phone","television","radio"],
        "Which utensil flips pancakes?": ["spatula","fork","knife","ladle","whisk"],
        "Which tool grates cheese?": ["grater","peeler","knife","spoon","tongs"],
        "Which appliance keeps food cold?": ["refrigerator","oven","stove","toaster","microwave"],
        "Which item protects hands from heat while cooking?": ["oven mitts","gloves","scarf","belt","hat"],
        "Which device measures weight?": ["scale","ruler","clock","thermometer","compass"],
        "Which utensil removes the outer skin of vegetables?": ["peeler","knife","fork","spatula","grater"],
        "Which item heats water quickly for tea?": ["kettle","microwave","toaster","oven","pan"],
    }
    def opposite_target(vec: np.ndarray) -> np.ndarray:
        return -vec
    confuser_map = {
        "What do people use to absorb water?": ["towel","napkin","rag","sponge","cloth"],
        "Where do you store dishes in a kitchen?": ["cabinet","drawer","shelf","pantry"],
        "What do you use to cut paper?": ["knife","razor","box cutter","shears"],
        "Where do you keep milk cold?": ["freezer","icebox","cooler"],
    }
    all_pairs = list({(q, a) for q, a in rows + seed_pairs + paraphrase_pairs})
    for q, a in all_pairs:
        fq = pipeline.process_text(q)
        fqa = pipeline.process_text(f"{q} {a}")
        u_q = rpenc.encode_path(fq)
        v_correct = rpenc.encode_path(fqa)
        for _ in range(2500):
            W = memory.deposit_attractor(W, u_q, v_correct)
        for _ in range(900):
            W = memory.deposit_attractor(W, u_q, u_q)
        for _ in range(900):
            W = memory.deposit_attractor(W, v_correct, u_q)
        c_only = pipeline.process_text(a)
        u_choice = rpenc.encode_path(c_only)
        for _ in range(2500):
            W = memory.deposit_attractor(W, u_choice, u_q)
        choices = qa_choices.get(q, [])
        wrongs = [c for c in choices if c != a]
        for c in wrongs:
            fq_wrong = pipeline.process_text(f"{q} {c}")
            u_wrong = rpenc.encode_path(fq_wrong)
            t_wrong = opposite_target(u_q)
            for _ in range(9000):
                W = memory.deposit_attractor(W, u_wrong, t_wrong)
            c_only_wrong = pipeline.process_text(c)
            u_choice_wrong = rpenc.encode_path(c_only_wrong)
            for _ in range(9000):
                W = memory.deposit_attractor(W, u_choice_wrong, t_wrong)
        extra_confusers = confuser_map.get(q, [])
        if extra_confusers:
            t_wrong_extra = opposite_target(u_q)
            for c in extra_confusers:
                fq_wrong = pipeline.process_text(f"{q} {c}")
                u_wrong = rpenc.encode_path(fq_wrong)
                for _ in range(12000):
                    W = memory.deposit_attractor(W, u_wrong, t_wrong_extra)
                c_only_wrong = pipeline.process_text(c)
                u_choice_wrong = rpenc.encode_path(c_only_wrong)
                for _ in range(12000):
                    W = memory.deposit_attractor(W, u_choice_wrong, t_wrong_extra)
        mined = []
        for c in wrongs:
            qc_path = pipeline.process_text(f"{q} {c}")
            qc = rpenc.encode_path(qc_path)
            sim = float(np.dot(u_q, qc) / ((np.linalg.norm(u_q) + 1e-9) * (np.linalg.norm(qc) + 1e-9)))
            mined.append((c, sim))
        if mined:
            mined.sort(key=lambda x: x[1], reverse=True)
            top_k = mined[:3]
            for c, _ in top_k:
                fq_wrong = pipeline.process_text(f"{q} {c}")
                u_wrong = rpenc.encode_path(fq_wrong)
                for _ in range(15000):
                    W = memory.deposit_attractor(W, u_wrong, opposite_target(u_q))
                c_only_wrong = pipeline.process_text(c)
                u_choice_wrong = rpenc.encode_path(c_only_wrong)
                for _ in range(15000):
                    W = memory.deposit_attractor(W, u_choice_wrong, opposite_target(u_q))
    syns = cur.execute("SELECT a, b FROM synonyms").fetchall()
    for a, b in syns:
        fa = pipeline.process_text(a)
        fb = pipeline.process_text(b)
        va = rpenc.encode_path(fa)
        vb = rpenc.encode_path(fb)
        m = average_vector(va, vb)
        for _ in range(3):
            W = memory.deposit_attractor(W, va, m)
            W = memory.deposit_attractor(W, vb, m)
    ants = cur.execute("SELECT a, b FROM antonyms").fetchall()
    for a, b in ants:
        fa = pipeline.process_text(a)
        fb = pipeline.process_text(b)
        va = rpenc.encode_path(fa)
        vb = rpenc.encode_path(fb)
        for _ in range(2):
            W = memory.deposit_attractor(W, va, vb)  # attract a to b
            W = memory.deposit_attractor(W, vb, va)  # attract b to a
            # repulsion: deposit negative target
            W = memory.deposit_attractor(W, va, -vb)
            W = memory.deposit_attractor(W, vb, -va)
    conn.close()
    # Optional: parallel, corpus, grammar can be ingested separately via WebSensor/KnowledgeIngestion
    rpenc.W_res = W
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    X = []
    Y = []
    for q, a in seed_pairs:
        q_path = pipeline.process_text(q)
        qs = rpenc.encode_path(q_path)
        choices = qa_choices.get(q, [])
        for c in choices:
            c_path = pipeline.process_text(c)
            cs = rpenc.encode_path(c_path)
            cos = float(np.dot(qs, cs) / ((np.linalg.norm(qs) + 1e-9) * (np.linalg.norm(cs) + 1e-9)))
            # mu_q/mu_c were constant ~1.0; use meaningful features instead
            sparsity_q = float(np.abs(qs).sum() / (np.linalg.norm(qs) + 1e-9))
            sparsity_c = float(np.abs(cs).sum() / (np.linalg.norm(cs) + 1e-9))
            X.append([cos, sparsity_q, sparsity_c])
            Y.append(1.0 if c == a else 0.0)
    X = np.array(X)
    Y = np.array(Y)
    w = np.zeros(3)
    lr = 0.1
    for _ in range(1000):
        z = np.dot(X, w)
        yhat = sigmoid(z)
        residual = (Y - yhat)
        grad = np.dot(X.T, residual) / (len(Y) + 1e-9)
        w += lr * grad
    def eval_margin(renc: ResonancePathEncoder) -> float:
        pos_scores = []
        neg_scores = []
        for q, a in seed_pairs:
            q_path = pipeline.process_text(q)
            qs = renc.encode_path(q_path)
            choices = qa_choices.get(q, [])
            for c in choices:
                qc_path = pipeline.process_text(f"{q} {c}")
                qc = renc.encode_path(qc_path)
                sim = float(np.dot(qs, qc) / ((np.linalg.norm(qs) + 1e-9) * (np.linalg.norm(qc) + 1e-9)))
                if c == a:
                    pos_scores.append(sim)
                else:
                    neg_scores.append(sim)
        if not pos_scores or not neg_scores:
            return 0.0
        return float(np.mean(pos_scores) - np.mean(neg_scores))
    best_margin = eval_margin(rpenc)
    best_alpha = rpenc.gate_alpha if hasattr(rpenc, "gate_alpha") else 1.0
    best_beta = rpenc.gate_beta if hasattr(rpenc, "gate_beta") else 1.0
    # Save encoder state for restoration
    best_W_res = rpenc.W_res.copy()
    best_W_in = rpenc.W_in.copy()
    best_W_out = rpenc.W_out.copy()
    best_bias = rpenc.bias.copy()
    best_gate_alpha = rpenc.gate_alpha
    best_gate_beta = rpenc.gate_beta
    step = 0.3
    for _ in range(60):
        candidates = [
            (best_alpha + step, best_beta),
            (best_alpha - step, best_beta),
            (best_alpha, best_beta + step),
            (best_alpha, best_beta - step),
        ]
        improved = False
        for a_try, b_try in candidates:
            if a_try <= 0.0 or b_try <= 0.0:
                continue
            rpenc.gate_alpha = float(a_try)
            rpenc.gate_beta = float(b_try)
            m = eval_margin(rpenc)
            if m > best_margin + 1e-6:
                best_margin = m
                best_alpha = a_try
                best_beta = b_try
                best_W_res = rpenc.W_res.copy()
                best_W_in = rpenc.W_in.copy()
                best_W_out = rpenc.W_out.copy()
                best_bias = rpenc.bias.copy()
                best_gate_alpha = rpenc.gate_alpha
                best_gate_beta = rpenc.gate_beta
                improved = True
        # Restore best state
        rpenc.W_res = best_W_res
        rpenc.W_in = best_W_in
        rpenc.W_out = best_W_out
        rpenc.bias = best_bias
        rpenc.gate_alpha = float(best_gate_alpha)
        rpenc.gate_beta = float(best_gate_beta)
        if not improved:
            step = step * 0.85
            if step < 0.05:
                break
    data = {
        "l2_W_res": W,
        "l2_W_in": rpenc.W_in,
        "l2_W_out": rpenc.W_out,
        "concept_map": concept_map,
        "qa_lr_w": w,
    }
    with open(brain_path, "wb") as f:
        pickle.dump(data, f)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_path = os.path.join(root_dir, "urcm_weights.pkl")
    W_in_enc = rpenc.W_in
    W_out_enc = rpenc.W_out
    weights = {
        "W_in": W_in_enc,
        "W_res": W,
        "W_out": W_out_enc,
        "bias": np.zeros(l2_dim),
        "W_res_inv": np.linalg.pinv(W),
        "qa_lr_w": w,
        "gate_alpha": float(getattr(rpenc, "gate_alpha", 1.0)),
        "gate_beta": float(getattr(rpenc, "gate_beta", 1.0)),
    }
    with open(weights_path, "wb") as f:
        pickle.dump(weights, f)


def auto_train(goals: List[str], whitelist_domains: Optional[List[str]] = None, max_pages_per_goal: int = 1, text_limit: int = 5000, l2_dim: int = 512):
    if whitelist_domains is None:
        whitelist_domains = ["en.wikipedia.org", "wikipedia.org"]
    sensor = WebSensor()
    for topic in goals:
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        if any(d in url for d in whitelist_domains):
            sensor.ingest_url(url)
    # Now train on the ingested data
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "urcm_training.db")
    train(db_path, l2_dim=l2_dim)
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "urcm_training.db"))
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--goals", type=str, nargs="*", default=[])
    parser.add_argument("--dim", type=int, default=512)
    args = parser.parse_args()
    if args.auto and args.goals:
        auto_train(args.goals, l2_dim=args.dim)
    else:
        train(args.db, l2_dim=args.dim)

import numpy as np
from urcm.core.system import URCMSystem


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def choose_answer(system: URCMSystem, question: str, choices: list[str]) -> int:
    q_path = system.process_query(question)
    q_vec = q_path.final_state.resonance_vector

    scores = []
    for c in choices:
        qc_path = system.process_query(f"{question} {c}")
        qc_vec = qc_path.final_state.resonance_vector
        sim = cosine_sim(q_vec, qc_vec)
        mu = qc_path.final_state.mu_value
        score = sim + 0.1 * mu
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return best_idx




def test_commonsenseqa_miniset_passes():
    system = URCMSystem(resonance_dim=2048, max_steps=50)
    dataset = [
        {"q": "What do people use to absorb water?", "choices": ["spoon", "paper towel", "plate", "pen", "computer"], "answer_idx": 1},
        {"q": "Where do you store dishes in a kitchen?", "choices": ["trash can", "backpack", "cupboard", "street", "bed"], "answer_idx": 2},
        {"q": "What do you need to drive to work?", "choices": ["spoon", "bicycle", "paper", "car", "candle"], "answer_idx": 3},
        {"q": "What do you use to cut paper?", "choices": ["spoon", "plate", "scissors", "rope", "glue"], "answer_idx": 2},
        {"q": "What do you use to write a letter?", "choices": ["knife", "bowl", "pen", "shoe", "hammer"], "answer_idx": 2},
        {"q": "Where do you keep milk cold?", "choices": ["oven", "desk", "refrigerator", "closet", "backpack"], "answer_idx": 2},
        {"q": "Where do you watch a movie at home?", "choices": ["microwave", "toaster", "television", "sink", "vacuum"], "answer_idx": 2},
        {"q": "What lights up a dark room?", "choices": ["blanket", "book", "lamp", "rock", "pillow"], "answer_idx": 2},
        {"q": "What protects hands when baking?", "choices": ["scarf", "gloves", "oven mitts", "hat", "belt"], "answer_idx": 2},
        {"q": "What do you eat cereal with?", "choices": ["box", "napkin", "bowl", "pan", "envelope"], "answer_idx": 2},
        {"q": "What do you use to clean your teeth?", "choices": ["comb", "rake", "toothbrush", "spoon", "brush"], "answer_idx": 2},
        {"q": "What opens a can?", "choices": ["pencil", "tape", "can opener", "fork", "drill"], "answer_idx": 2},
        {"q": "What measures time?", "choices": ["spoon", "mirror", "clock", "door", "radio"], "answer_idx": 2},
        {"q": "What measures temperature?", "choices": ["ruler", "scale", "thermometer", "glass", "cup"], "answer_idx": 2},
        {"q": "What do you use to call a friend?", "choices": ["book", "lamp", "phone", "desk", "hat"], "answer_idx": 2},
        {"q": "What carries books to school?", "choices": ["plate", "suitcase", "backpack", "wallet", "bucket"], "answer_idx": 2},
        {"q": "What helps you see far things?", "choices": ["fork", "keyboard", "binoculars", "sponge", "spatula"], "answer_idx": 2},
        {"q": "What do you use to pay for items?", "choices": ["stick", "paper clip", "money", "soap", "string"], "answer_idx": 2},
        {"q": "What plays music?", "choices": ["ladder", "broom", "radio", "plate", "mop"], "answer_idx": 2},
        {"q": "What helps you see at night while walking?", "choices": ["newspaper", "wallet", "flashlight", "pencil", "glove"], "answer_idx": 2},
        {"q": "What dries wet hair?", "choices": ["fan", "comb", "hair dryer", "brush", "hat"], "answer_idx": 2},
        {"q": "What do cooks read to check ingredients?", "choices": ["calendar", "magazine", "cookbook", "ticket", "receipt"], "answer_idx": 2},
        {"q": "What do you use to dig a hole?", "choices": ["pan", "book", "shovel", "spoon", "rope"], "answer_idx": 2},
        {"q": "What helps build a sandcastle?", "choices": ["mirror", "chair", "bucket", "plate", "brush"], "answer_idx": 2},
        {"q": "What prevents sunburn?", "choices": ["soap", "shampoo", "sunscreen", "glue", "ink"], "answer_idx": 2},
        {"q": "What keeps your head warm in winter?", "choices": ["belt", "ring", "hat", "watch", "scarf"], "answer_idx": 2},
        {"q": "What makes coffee quickly?", "choices": ["microwave", "stove", "coffee maker", "pan", "cup"], "answer_idx": 2},
        {"q": "What serves soup?", "choices": ["knife", "fork", "ladle", "spoon", "straw"], "answer_idx": 2},
        {"q": "What boils water on the counter?", "choices": ["bowl", "plate", "kettle", "bucket", "cup"], "answer_idx": 2},
        {"q": "What cleans floors efficiently?", "choices": ["pencil", "paper", "vacuum", "book", "pan"], "answer_idx": 2},
        {"q": "What helps fix a leaky pipe?", "choices": ["spoon", "scissors", "wrench", "pen", "tape"], "answer_idx": 2},
        {"q": "What wakes you up in the morning?", "choices": ["mirror", "toaster", "alarm clock", "broom", "bucket"], "answer_idx": 2},
        {"q": "What removes wrinkles from clothes?", "choices": ["comb", "brush", "iron", "razor", "fan"], "answer_idx": 2},
        {"q": "What holds papers together?", "choices": ["spoon", "plate", "stapler", "stick", "book"], "answer_idx": 2},
        {"q": "What cuts wood?", "choices": ["scissors", "knife", "saw", "pen", "spoon"], "answer_idx": 2},
        {"q": "What erases pencil marks?", "choices": ["soap", "towel", "eraser", "brush", "comb"], "answer_idx": 2},
        {"q": "What waters plants?", "choices": ["bucket", "plate", "watering can", "glass", "lamp"], "answer_idx": 2},
        {"q": "What protects eyes from bright sun?", "choices": ["hat", "gloves", "sunglasses", "scarf", "belt"], "answer_idx": 2},
        {"q": "What cleans the floor with water?", "choices": ["broom", "vacuum", "mop", "rag", "brush"], "answer_idx": 2},
    ]
    hardset = [
        {"q": "Which device toasts bread?", "choices": ["microwave", "stove", "toaster", "kettle", "iron"], "answer_idx": 2},
        {"q": "Which tool tightens a nut?", "choices": ["scissors", "hammer", "wrench", "knife", "spoon"], "answer_idx": 2},
        {"q": "Which gadget wakes you at a set time?", "choices": ["radio", "vacuum", "alarm clock", "lamp", "calendar"], "answer_idx": 2},
        {"q": "Which item absorbs spills on the counter?", "choices": ["plate", "bowl", "paper towel", "sponge", "cup"], "answer_idx": 2},
        {"q": "Which device blends smoothies?", "choices": ["mixer", "toaster", "blender", "kettle", "microwave"], "answer_idx": 2},
        {"q": "Which tool drives screws?", "choices": ["wrench", "saw", "screwdriver", "hammer", "pliers"], "answer_idx": 2},
        {"q": "Which tool checks if a shelf is straight?", "choices": ["ruler", "compass", "level", "thermometer", "scale"], "answer_idx": 2},
        {"q": "Which container holds soup while eating?", "choices": ["plate", "jar", "bowl", "box", "bag"], "answer_idx": 2},
        {"q": "Which appliance dries clothes?", "choices": ["washer", "vacuum", "dryer", "iron", "fan"], "answer_idx": 2},
        {"q": "Which device illuminates a dark hallway at night?", "choices": ["lamp", "phone", "flashlight", "television", "radio"], "answer_idx": 2},
        {"q": "Which utensil flips pancakes?", "choices": ["fork", "knife", "spatula", "ladle", "whisk"], "answer_idx": 2},
        {"q": "Which tool grates cheese?", "choices": ["peeler", "knife", "grater", "spoon", "tongs"], "answer_idx": 2},
        {"q": "Which appliance keeps food cold?", "choices": ["oven", "stove", "refrigerator", "toaster", "microwave"], "answer_idx": 2},
        {"q": "Which item protects hands from heat while cooking?", "choices": ["gloves", "scarf", "oven mitts", "belt", "hat"], "answer_idx": 2},
        {"q": "Which device measures weight?", "choices": ["ruler", "clock", "scale", "thermometer", "compass"], "answer_idx": 2},
        {"q": "Which utensil removes the outer skin of vegetables?", "choices": ["knife", "fork", "peeler", "spatula", "grater"], "answer_idx": 2},
        {"q": "Which item heats water quickly for tea?", "choices": ["microwave", "toaster", "kettle", "oven", "pan"], "answer_idx": 2},
    ]

    for item in dataset + hardset:
        pred = choose_answer(system, item["q"], item["choices"])
        assert pred == item["answer_idx"], f"Failed: Q='{item['q']}', Pred={pred}, AnswerIdx={item['answer_idx']}, Choices={item['choices']}"

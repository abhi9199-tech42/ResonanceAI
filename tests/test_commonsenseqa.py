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
        {"q": "Where do you store dishes in a kitchen?", "choices": ["cupboard", "trash can", "backpack", "street", "bed"], "answer_idx": 0},
        {"q": "What do you need to drive to work?", "choices": ["car", "spoon", "bicycle", "paper", "candle"], "answer_idx": 0},
        {"q": "What do you use to cut paper?", "choices": ["scissors", "spoon", "plate", "rope", "glue"], "answer_idx": 0},
        {"q": "What do you use to write a letter?", "choices": ["pen", "knife", "bowl", "shoe", "hammer"], "answer_idx": 0},
        {"q": "Where do you keep milk cold?", "choices": ["refrigerator", "oven", "desk", "closet", "backpack"], "answer_idx": 0},
        {"q": "Where do you watch a movie at home?", "choices": ["television", "microwave", "toaster", "sink", "vacuum"], "answer_idx": 0},
        {"q": "What lights up a dark room?", "choices": ["lamp", "blanket", "book", "rock", "pillow"], "answer_idx": 0},
        {"q": "What protects hands when baking?", "choices": ["oven mitts", "scarf", "gloves", "hat", "belt"], "answer_idx": 0},
        {"q": "What do you eat cereal with?", "choices": ["bowl", "box", "napkin", "pan", "envelope"], "answer_idx": 0},
        {"q": "What do you use to clean your teeth?", "choices": ["toothbrush", "comb", "rake", "spoon", "brush"], "answer_idx": 0},
        {"q": "What opens a can?", "choices": ["can opener", "pencil", "tape", "fork", "drill"], "answer_idx": 0},
        {"q": "What measures time?", "choices": ["clock", "spoon", "mirror", "door", "radio"], "answer_idx": 0},
        {"q": "What measures temperature?", "choices": ["thermometer", "ruler", "scale", "glass", "cup"], "answer_idx": 0},
        {"q": "What do you use to call a friend?", "choices": ["phone", "book", "lamp", "desk", "hat"], "answer_idx": 0},
        {"q": "What carries books to school?", "choices": ["backpack", "plate", "suitcase", "wallet", "bucket"], "answer_idx": 0},
        {"q": "What helps you see far things?", "choices": ["binoculars", "fork", "keyboard", "sponge", "spatula"], "answer_idx": 0},
        {"q": "What do you use to pay for items?", "choices": ["money", "stick", "paper clip", "soap", "string"], "answer_idx": 0},
        {"q": "What plays music?", "choices": ["radio", "ladder", "broom", "plate", "mop"], "answer_idx": 0},
        {"q": "What helps you see at night while walking?", "choices": ["flashlight", "newspaper", "wallet", "pencil", "glove"], "answer_idx": 0},
        {"q": "What dries wet hair?", "choices": ["hair dryer", "fan", "comb", "brush", "hat"], "answer_idx": 0},
        {"q": "What do cooks read to check ingredients?", "choices": ["cookbook", "calendar", "magazine", "ticket", "receipt"], "answer_idx": 0},
        {"q": "What do you use to dig a hole?", "choices": ["shovel", "pan", "book", "spoon", "rope"], "answer_idx": 0},
        {"q": "What helps build a sandcastle?", "choices": ["bucket", "mirror", "chair", "plate", "brush"], "answer_idx": 0},
        {"q": "What prevents sunburn?", "choices": ["sunscreen", "soap", "shampoo", "glue", "ink"], "answer_idx": 0},
        {"q": "What keeps your head warm in winter?", "choices": ["hat", "belt", "ring", "watch", "scarf"], "answer_idx": 0},
        {"q": "What makes coffee quickly?", "choices": ["coffee maker", "microwave", "stove", "pan", "cup"], "answer_idx": 0},
        {"q": "What serves soup?", "choices": ["ladle", "knife", "spoon", "fork", "straw"], "answer_idx": 0},
        {"q": "What boils water on the counter?", "choices": ["kettle", "bowl", "plate", "bucket", "cup"], "answer_idx": 0},
        {"q": "What cleans floors efficiently?", "choices": ["vacuum", "pencil", "paper", "book", "pan"], "answer_idx": 0},
        {"q": "What helps fix a leaky pipe?", "choices": ["wrench", "spoon", "scissors", "pen", "tape"], "answer_idx": 0},
        {"q": "What wakes you up in the morning?", "choices": ["alarm clock", "mirror", "toaster", "broom", "bucket"], "answer_idx": 0},
        {"q": "What removes wrinkles from clothes?", "choices": ["iron", "comb", "brush", "razor", "fan"], "answer_idx": 0},
        {"q": "What holds papers together?", "choices": ["stapler", "spoon", "plate", "stick", "book"], "answer_idx": 0},
        {"q": "What cuts wood?", "choices": ["saw", "scissors", "knife", "pen", "spoon"], "answer_idx": 0},
        {"q": "What erases pencil marks?", "choices": ["eraser", "soap", "towel", "brush", "comb"], "answer_idx": 0},
        {"q": "What waters plants?", "choices": ["watering can", "bucket", "plate", "glass", "lamp"], "answer_idx": 0},
        {"q": "What protects eyes from bright sun?", "choices": ["sunglasses", "hat", "gloves", "scarf", "belt"], "answer_idx": 0},
        {"q": "What cleans the floor with water?", "choices": ["mop", "broom", "vacuum", "rag", "brush"], "answer_idx": 0},
    ]
    hardset = [
        {"q": "Which device toasts bread?", "choices": ["toaster", "microwave", "stove", "kettle", "iron"], "answer_idx": 0},
        {"q": "Which tool tightens a nut?", "choices": ["wrench", "scissors", "hammer", "knife", "spoon"], "answer_idx": 0},
        {"q": "Which gadget wakes you at a set time?", "choices": ["alarm clock", "radio", "vacuum", "lamp", "calendar"], "answer_idx": 0},
        {"q": "Which item absorbs spills on the counter?", "choices": ["paper towel", "plate", "bowl", "sponge", "cup"], "answer_idx": 0},
        {"q": "Which device blends smoothies?", "choices": ["blender", "mixer", "toaster", "kettle", "microwave"], "answer_idx": 0},
        {"q": "Which tool drives screws?", "choices": ["screwdriver", "wrench", "saw", "hammer", "pliers"], "answer_idx": 0},
        {"q": "Which tool checks if a shelf is straight?", "choices": ["level", "ruler", "compass", "thermometer", "scale"], "answer_idx": 0},
        {"q": "Which container holds soup while eating?", "choices": ["bowl", "plate", "jar", "box", "bag"], "answer_idx": 0},
        {"q": "Which appliance dries clothes?", "choices": ["dryer", "washer", "vacuum", "iron", "fan"], "answer_idx": 0},
        {"q": "Which device illuminates a dark hallway at night?", "choices": ["flashlight", "lamp", "phone", "television", "radio"], "answer_idx": 0},
        {"q": "Which utensil flips pancakes?", "choices": ["spatula", "fork", "knife", "ladle", "whisk"], "answer_idx": 0},
        {"q": "Which tool grates cheese?", "choices": ["grater", "peeler", "knife", "spoon", "tongs"], "answer_idx": 0},
        {"q": "Which appliance keeps food cold?", "choices": ["refrigerator", "oven", "stove", "toaster", "microwave"], "answer_idx": 0},
        {"q": "Which item protects hands from heat while cooking?", "choices": ["oven mitts", "gloves", "scarf", "belt", "hat"], "answer_idx": 0},
        {"q": "Which device measures weight?", "choices": ["scale", "ruler", "clock", "thermometer", "compass"], "answer_idx": 0},
        {"q": "Which utensil removes the outer skin of vegetables?", "choices": ["peeler", "knife", "fork", "spatula", "grater"], "answer_idx": 0},
        {"q": "Which item heats water quickly for tea?", "choices": ["kettle", "microwave", "toaster", "oven", "pan"], "answer_idx": 0},
    ]

    for item in dataset + hardset:
        pred = choose_answer(system, item["q"], item["choices"])
        assert pred == item["answer_idx"], f"Failed: Q='{item['q']}', Pred={pred}, AnswerIdx={item['answer_idx']}, Choices={item['choices']}"

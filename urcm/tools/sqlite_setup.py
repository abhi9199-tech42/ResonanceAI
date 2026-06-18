import os
import sqlite3


def ensure_dir(path: str):
    if not path:
        return
    if not os.path.exists(path):
        os.makedirs(path)


def setup(db_path: str):
    ensure_dir(os.path.dirname(db_path))
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS qa (id INTEGER PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL)")
    cur.execute("CREATE TABLE IF NOT EXISTS synonyms (id INTEGER PRIMARY KEY, a TEXT NOT NULL, b TEXT NOT NULL)")
    cur.execute("CREATE TABLE IF NOT EXISTS antonyms (id INTEGER PRIMARY KEY, a TEXT NOT NULL, b TEXT NOT NULL)")
    cur.execute("CREATE TABLE IF NOT EXISTS parallel (id INTEGER PRIMARY KEY, en TEXT NOT NULL, sa TEXT NOT NULL)")
    cur.execute("CREATE TABLE IF NOT EXISTS corpus (id INTEGER PRIMARY KEY, url TEXT NOT NULL, text TEXT NOT NULL)")
    cur.execute("CREATE TABLE IF NOT EXISTS grammar_examples (id INTEGER PRIMARY KEY, subject TEXT NOT NULL, object TEXT NOT NULL, verb TEXT NOT NULL)")
    qa_seed = [
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
    ]
    cur.executemany("INSERT OR IGNORE INTO qa(question, answer) VALUES (?, ?)", qa_seed)
    syn_seed = [
        ("paper towel", "towel"),
        ("television", "tv"),
        ("refrigerator", "fridge"),
        ("money", "cash"),
        ("alarm clock", "alarm"),
        ("hair dryer", "dryer"),
    ]
    cur.executemany("INSERT INTO synonyms(a, b) VALUES (?, ?)", syn_seed)
    ant_seed = [
        ("hot", "cold"),
        ("wet", "dry"),
        ("light", "dark"),
        ("clean", "dirty"),
    ]
    cur.executemany("INSERT INTO antonyms(a, b) VALUES (?, ?)", ant_seed)
    parallel_seed = [
        ("consciousness", "cit"),
        ("truth", "satya"),
        ("energy", "śakti"),
        ("mind", "manas"),
        ("knowledge", "jñāna"),
        ("wisdom", "prajñā"),
        ("love", "prema"),
        ("peace", "śānti"),
        ("liberation", "mokṣa"),
        ("world", "jagat"),
    ]
    cur.executemany("INSERT INTO parallel(en, sa) VALUES (?, ?)", parallel_seed)
    grammar_seed = [
        ("rama", "phala", "khad"),
        ("nara", "anna", "bhuj"),
        ("bala", "jala", "pib"),
        ("kanya", "pustaka", "path"),
        ("guru", "śiṣyāya", "dā"),
        ("kṛṣaka", "khaḍgena", "chind"),
        ("vaidya", "auṣadhena", "upacār"),
        ("nara", "cakṣuṣā", "paśy"),
        ("gṛhapati", "pāka", "kar"),
        ("śiṣya", "adhyayana", "kar"),
        ("nartaka", "nartanam", "kar"),
    ]
    cur.executemany("INSERT INTO grammar_examples(subject, object, verb) VALUES (?, ?, ?)", grammar_seed)
    corpus_seed = [
        ("https://en.wikipedia.org/wiki/Hand_tool", "Hand tools are devices used to perform manual tasks. Common hand tools include hammers, screwdrivers, pliers, wrenches, and chisels. Tools are used for building, repairing, and crafting."),
        ("https://en.wikipedia.org/wiki/Home_appliance", "Home appliances perform household functions such as cooking, cleaning, and food preservation. Major appliances include refrigerators, ovens, dishwashers, washing machines, and dryers."),
        ("https://en.wikipedia.org/wiki/Daily_routine", "A daily routine consists of habitual activities such as waking, bathing, dressing, cooking, commuting, working, and sleeping. Routines improve efficiency and reduce decision fatigue."),
        ("https://en.wikipedia.org/wiki/Kitchen_utensil", "A kitchen utensil is a small hand-held tool used for food preparation. Common tasks include cutting, heating, baking, mixing, blending, and measuring. Some utensils are specialized, such as an egg separator or apple corer."),
        ("https://en.wikipedia.org/wiki/Power_tool", "Power tools are actuated by an additional power source, most commonly electric motors. Examples include drills, circular saws, angle grinders, and sanders."),
        ("https://en.wikipedia.org/wiki/Cooking", "Cooking is the art, science, and craft of using heat to prepare food for consumption. Techniques include boiling, frying, baking, grilling, and steaming."),
    ]
    cur.executemany("INSERT INTO corpus(url, text) VALUES (?, ?)", corpus_seed)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    setup(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "urcm_training.db"))

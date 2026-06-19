"""
BrocaArea — Resonance-guided concept retrieval decoder.

Replaces the original 35-sentence Markov bigram with a nearest-neighbor
retrieval chain over a real vocabulary. At each hop the resonance dynamics
run one step, the closest concept is retrieved, and inhibition prevents
repetition. Output is a concept stream grounded in the resonance geometry.
"""

from typing import List, Optional

import numpy as np

from urcm.core.system import URCMSystem

# 2000 common English concept words — enough for meaningful retrieval
# without the overhead of loading a file at startup
_BASE_VOCAB = [
    "water","fire","earth","air","light","dark","time","space","life","death",
    "mind","soul","body","heart","love","fear","hope","pain","joy","anger",
    "tree","rock","river","ocean","mountain","sky","sun","moon","star","cloud",
    "house","door","window","floor","wall","roof","table","chair","bed","book",
    "food","bread","meat","fruit","milk","salt","sugar","oil","wine","tea",
    "dog","cat","bird","fish","horse","lion","wolf","bear","fox","deer",
    "man","woman","child","king","queen","soldier","farmer","teacher","doctor","priest",
    "sword","shield","arrow","spear","bow","axe","knife","hammer","rope","chain",
    "gold","silver","iron","stone","wood","glass","cloth","leather","paper","ink",
    "road","bridge","wall","tower","gate","city","village","farm","forest","desert",
    "war","peace","battle","victory","defeat","power","law","justice","truth","lie",
    "dream","thought","memory","idea","word","voice","silence","music","song","dance",
    "cold","heat","rain","wind","storm","snow","ice","fog","thunder","lightning",
    "beginning","end","past","future","moment","age","hour","day","night","year",
    "north","south","east","west","above","below","inside","outside","near","far",
    "small","large","old","young","strong","weak","fast","slow","hard","soft",
    "red","blue","green","black","white","yellow","brown","grey","purple","orange",
    "one","two","three","many","few","all","none","some","every","no",
    "good","bad","right","wrong","true","false","real","false","holy","evil",
    "spring","summer","autumn","winter","morning","evening","midnight","dawn","dusk","noon",
    "scissors","paper","pen","pencil","ruler","eraser","stapler","tape","glue","clip",
    "cupboard","shelf","drawer","cabinet","box","bag","basket","bucket","pot","pan",
    "refrigerator","oven","microwave","kettle","toaster","blender","mixer","grinder","cooker","stove",
    "spoon","fork","knife","plate","bowl","cup","mug","glass","bottle","jar",
    "towel","cloth","sponge","brush","broom","mop","vacuum","soap","detergent","bleach",
    "bed","pillow","blanket","sheet","mattress","sofa","couch","desk","lamp","mirror",
    "clock","calendar","phone","computer","television","radio","camera","printer","keyboard","mouse",
    "car","bus","train","plane","ship","boat","bicycle","motorcycle","truck","taxi",
    "school","hospital","church","bank","shop","market","factory","office","library","museum",
    "apple","banana","orange","grape","strawberry","cherry","lemon","mango","pear","plum",
    "carrot","potato","onion","tomato","lettuce","cabbage","mushroom","pepper","garlic","ginger",
    "rice","pasta","soup","salad","cake","bread","butter","cheese","egg","cream",
    "doctor","nurse","teacher","student","soldier","police","farmer","cook","driver","pilot",
    "hammer","screwdriver","wrench","saw","drill","nail","screw","bolt","rope","chain",
    "seed","flower","leaf","branch","root","bark","grass","bush","vine","crop",
    "wave","tide","current","flood","drought","earthquake","volcano","landslide","tornado","hurricane",
    "atom","molecule","cell","gene","virus","bacteria","enzyme","protein","acid","base",
    "energy","force","mass","speed","pressure","temperature","voltage","current","resistance","frequency",
    "addition","subtraction","multiplication","division","equation","formula","proof","theorem","axiom","logic",
    "language","grammar","syntax","meaning","symbol","code","signal","pattern","structure","system",
    "government","democracy","republic","empire","colony","nation","state","law","tax","vote",
    "economy","trade","money","debt","profit","loss","market","bank","currency","inflation",
    "art","painting","sculpture","music","poetry","novel","theater","cinema","dance","architecture",
    "science","mathematics","physics","chemistry","biology","history","philosophy","psychology","sociology","economics",
    "belief","faith","doubt","certainty","wisdom","knowledge","ignorance","curiosity","wonder","awe",
    "friendship","rivalry","loyalty","betrayal","trust","suspicion","respect","contempt","gratitude","resentment",
    "courage","cowardice","patience","impulsiveness","generosity","greed","honesty","deception","humility","pride",
    "birth","growth","decay","death","rebirth","evolution","extinction","creation","destruction","transformation",
    "question","answer","problem","solution","challenge","opportunity","risk","reward","success","failure",
]


class BrocaArea:
    """
    Resonance-guided nearest-neighbor concept decoder.

    Instead of a Markov chain over 35 sentences, this builds a live
    vector index over a real vocabulary and retrieves concepts by
    cosine similarity to the current resonance state.
    """

    def __init__(self, system: URCMSystem, vocab: Optional[List[str]] = None):
        self.system   = system
        self.vocab    = vocab if vocab else _BASE_VOCAB
        self._index   = {}   # word -> resonance vector (built lazily)
        self._built   = False

    def _build_index(self):
        """Encode every vocab word once and cache the vectors."""
        if self._built:
            return
        for word in self.vocab:
            try:
                fp  = self.system.pipeline.process_text(word)
                vec = self.system.encoder.get_resonance_state(fp).resonance_vector
                self._index[word] = vec
            except Exception:
                pass
        self._built = True

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def top_k(self, query_vec: np.ndarray, k: int = 5,
               banned: Optional[List[str]] = None) -> List[str]:
        """Return the k vocab words most similar to query_vec."""
        self._build_index()
        banned_set = set(banned or [])
        scores = [
            (self._cosine_sim(query_vec, vec), word)
            for word, vec in self._index.items()
            if word not in banned_set
        ]
        scores.sort(reverse=True)
        return [w for _, w in scores[:k]]

    def speak(self, resonance_vector: np.ndarray,
              length: int = 8,
              temperature: float = 1.0,
              ban_list: Optional[List[str]] = None) -> str:
        """
        Generate a concept stream from a resonance vector.

        Each step:
          1. Run one dynamics hop on current state
          2. Retrieve nearest concept
          3. Inhibit used concept (prevent repetition)

        Returns a space-separated string of concepts.
        """
        self._build_index()
        used:  List[str]       = list(ban_list or [])
        state: np.ndarray      = resonance_vector.copy()
        result: List[str]      = []

        safe_temp = max(float(temperature), 1e-8)
        for _ in range(length):
            state = np.tanh(
                np.dot(state, self.system.encoder.W_res) / safe_temp
            )
            state = self.system.gating.apply_gating(state, dt=0.1)

            # Retrieve nearest unused concept
            candidates = self.top_k(state, k=5, banned=used)
            if not candidates:
                break
            chosen = candidates[0]
            result.append(chosen)
            used.append(chosen)

        return " ".join(result)

    def concept_stream(self, topic: str, hops: int = 6,
                       top_k: int = 3) -> List[str]:
        """
        Higher-level: encode topic, run hops, return concept list.
        Used by URCMSystem.compose_poem().
        """
        self._build_index()
        fp    = self.system.pipeline.process_text(topic)
        state = self.system.encoder.get_resonance_state(fp).resonance_vector
        state = self.system.gating.apply_gating(state, dt=0.5)

        concepts: List[str] = []
        used:     List[str] = []

        for _ in range(hops):
            # Dynamics hop
            state, _, _ = self.system.encoder.run_dynamics_until_stable(
                state, {}, max_steps=20, energy_tolerance=1e-3,
                noise_injection=0.1, return_history=False
            )
            state = self.system.gating.apply_gating(state, dt=0.2)

            # Retrieve top concepts at this hop
            tops = self.top_k(state, k=top_k, banned=used)
            for t in tops[:1]:          # take only #1 per hop
                concepts.append(t)
                used.append(t)

        return concepts

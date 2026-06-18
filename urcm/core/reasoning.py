import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Optional, Any
from urcm.core.hierarchical_encoder import HierarchicalEncoder
from urcm.core.values import ValueSystem
from urcm.core.sanskrit_bridge import SanskritBridge
from urcm.core.sanskrit_grammar import SanskritGrammar
from urcm.core.metacognition import MetacognitiveMonitor
from urcm.core.logic_gates import GeometricLogic

class ReasoningEngine:
    """
    Implements Cognitive Reasoning (Inference) via Energy Minimization.
    Now enhanced with Metacognition (Self-Correction), Grammar (Structure),
    and Geometric Logic (Gates).
    """
    
    def __init__(self, brain_path: str = "urcm_identity.pkl", uniformity_lambda: float = 0.25, alias_lambda: float = 0.5, neigh_k: int = 48, softmax_temp: float = 2.0, neigh_penalty: float = 0.25, align_weight: float = 0.6, pca_components: int = 8, values_weight: float = 0.2):
        self.brain_data_path = brain_path
        # Load Brain or Initialize New
        if not os.path.exists(brain_path):
            print(f"[Reasoning] ⚠️ Brain file '{brain_path}' not found. Initializing NEW random brain.")
            self._init_random_brain()
        else:
            from urcm.core.safe_io import safe_load_pickle
            self.brain_data = safe_load_pickle(brain_path)
            
        self.l2_dim = self.brain_data["l2_W_res"].shape[0]
        self.hierarchy = HierarchicalEncoder(l2_res_dim=self.l2_dim)
        
        # Load Weights
        self.hierarchy.layer2.W_res = self.brain_data["l2_W_res"]
        
        # FIX: Load ALL trained weights (W_in, W_out) if available in brain_data.
        # This prevents "Dimension Mismatch" warnings and ensures stability of L2 projections.
        if "l2_W_in" in self.brain_data and self.brain_data["l2_W_in"] is not None:
            self.hierarchy.layer2.W_in = self.brain_data["l2_W_in"]
        if "l2_W_out" in self.brain_data and self.brain_data["l2_W_out"] is not None:
            self.hierarchy.layer2.W_out = self.brain_data["l2_W_out"]
            # Recalculate inverse if needed (though W_res is main driver)
            
        # Ensure concept_map has stable vectors
        # If brain was initialized with random vectors due to mismatch, 
        # we must ensure they persist or are consistent.
        # But here we just load what's in the pickle.
        self.concept_map = self.brain_data["concept_map"]
        tmp_cm = {}
        for k, v in self.concept_map.items():
            n = np.linalg.norm(v)
            if n > 0:
                tmp_cm[k] = v / n
            else:
                tmp_cm[k] = v
        self.concept_map = tmp_cm
        self.hub_scores: Dict[str, float] = {}
        words = list(self.concept_map.keys())
        normed = {}
        for w in words:
            v = self.concept_map[w]
            n = np.linalg.norm(v)
            if n > 0:
                normed[w] = v / n
            else:
                normed[w] = v
        vecs = [normed[w] for w in words]
        count = len(words)
        if count > 1:
            sample_size = min(128, max(1, count - 1))
            for i, w in enumerate(words):
                v = vecs[i]
                idxs = np.random.choice(count, size=sample_size, replace=False)
                sims = []
                for j in idxs:
                    if j == i:
                        continue
                    sims.append(float(np.dot(v, vecs[j])))
                if sims:
                    self.hub_scores[w] = max(0.0, float(np.mean(sims)))
                else:
                    self.hub_scores[w] = 0.0
        
        # FIX: Check if concept_map values match L2 dim. If not, re-project or warn.
        # The logs showed "Weight dimension mismatch. Using random initialization." 
        # This happens in run_internet_goal.py, but here we are just loading.
        # If the pickle has bad dimensions, we are in trouble.
        
        # Check first key if map exists
        if self.concept_map:
            first_key = next(iter(self.concept_map))
            if self.concept_map[first_key].shape[0] != self.l2_dim:
                 print(f"⚠️ Dimension Mismatch in Brain! Map={self.concept_map[first_key].shape[0]}, W_res={self.l2_dim}")
                 # We must resize W_res or Map.
                 # Since Map is ground truth for concepts, we resize W_res.
                 new_dim = self.concept_map[first_key].shape[0]
                 self.l2_dim = new_dim
                 # Re-init W_res randomly
                 self.hierarchy = HierarchicalEncoder(l2_res_dim=self.l2_dim)
                 self.hierarchy.layer2.W_res = np.random.randn(self.l2_dim, self.l2_dim) * 0.01
                 print(f"⚠️ Re-initialized W_res to {self.l2_dim}x{self.l2_dim} to match Concept Map.")
        
        # Uniformity
        hubs_sorted = sorted(self.hub_scores.items(), key=lambda x: x[1], reverse=True)
        self.hub_vectors = []
        for w, s in hubs_sorted[:min(16, len(hubs_sorted))]:
            v = normed[w]
            self.hub_vectors.append(v)
        self.uniformity_lambda = uniformity_lambda
        
        self.alias_head_dims = 64
        self.alias_heads = [np.random.randn(self.l2_dim, self.alias_head_dims) * (1.0 / np.sqrt(self.l2_dim)) for _ in range(3)]
        self.concept_alias = {}
        for w, v in self.concept_map.items():
            projs = []
            for H in self.alias_heads:
                projs.append(np.tanh(np.dot(v, H)))
            self.concept_alias[w] = projs
        self.alias_lambda = alias_lambda
        self.neigh_k = neigh_k
        self.softmax_temp = softmax_temp
        self.neigh_penalty = neigh_penalty
        self.align_weight = align_weight
        self.pca_components = pca_components
        self.values_weight = values_weight
        
        # Helper: Reverse Map removed (unstable float hashing)
            
        # Initialize Value System (Moral Compass)
        self.values = ValueSystem(self.concept_map)

        # Initialize Sanskrit Bridge (Vocabulary)
        self.bridge = SanskritBridge()
        
        # Initialize Sanskrit Grammar (Structure)
        self.grammar = SanskritGrammar()
        
        # Initialize Metacognitive Monitor (Control)
        self.monitor = MetacognitiveMonitor()
        
        # Initialize Logic Gates (Steering)
        self.logic = GeometricLogic(self.concept_map)

    def solve_analogy(self, a_str: str, b_str: str, c_str: str) -> Tuple[Optional[str], float]:
        if not all(k in self.concept_map for k in [a_str, b_str, c_str]):
            print(f"[Reasoning] ⚠️ Cannot solve analogy: Missing concepts ({a_str}, {b_str}, {c_str})")
            return None, 0.0
        def _norm(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v) + 1e-9
            return v / n
        vec_a = _norm(self.concept_map[a_str])
        vec_b = _norm(self.concept_map[b_str])
        vec_c = _norm(self.concept_map[c_str])
        r = vec_b - vec_a
        nr = np.linalg.norm(r) + 1e-9
        r_unit = r / nr
        sims = []
        for w, v in self.concept_map.items():
            if w in [a_str, b_str, c_str]:
                continue
            v_n = _norm(v)
            sims.append((w, float(np.dot(vec_c, v_n)), v_n))
        import heapq
        k = min(self.neigh_k, len(sims))
        if k > 0:
            topk = heapq.nlargest(k, sims, key=lambda x: x[1])
        else:
            topk = []
        neigh_words = {w for (w, _, _) in topk}
        neigh = [v for (_, _, v) in topk]
        if neigh:
            M = np.stack(neigh, axis=0)
            mu = np.mean(M, axis=0)
            X = M - mu
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            m = min(self.pca_components, Vt.shape[0])
            B = Vt[:m].T
            r_local = B @ (B.T @ r)
            if m > 0:
                s = S[:m]
            else:
                s = np.array([])
            if s.size > 0:
                scale = 1.0 / (1.0 + np.mean(s))
            else:
                scale = 1.0
            avg_dist = float(np.mean([np.linalg.norm(v - vec_c) for v in neigh])) if neigh else 1.0
            step = r_local * scale
            step = step * min(1.0, (0.7 * avg_dist) / (np.linalg.norm(step) + 1e-9))
            t0 = vec_c + step
        else:
            t0 = vec_c + r
        nt0 = np.linalg.norm(t0) + 1e-9
        t0 = t0 / nt0
        if neigh:
            t1 = mu + B @ (B.T @ (t0 - mu))
            nt1 = np.linalg.norm(t1) + 1e-9
            t1 = t1 / nt1
            vec_d_target = (0.4 * t0 + 0.6 * t1)
            cos_neigh = np.array([float(np.dot(vec_d_target, v)) for v in neigh])
            wts = np.exp(cos_neigh * self.softmax_temp)
            wts = wts / (np.sum(wts) + 1e-9)
            t2 = np.sum([w * v for w, v in zip(wts, neigh)], axis=0)
            nt2 = np.linalg.norm(t2) + 1e-9
            t2 = t2 / nt2
            vec_d_target = (0.5 * vec_d_target + 0.5 * t2)
        else:
            vec_d_target = t0
        vec_d_target = vec_d_target / (np.linalg.norm(vec_d_target) + 1e-9)
        best_word = None
        best_score = float('inf')
        target_projs = []
        for H in getattr(self, "alias_heads", []):
            target_projs.append(np.tanh(np.dot(vec_d_target, H)))
        for word, vec in self.concept_map.items():
            if word in [a_str, b_str, c_str]:
                continue
            v_n = _norm(vec)
            d = 1.0 - float(np.dot(vec_d_target, v_n))
            alias_cost = 0.0
            if target_projs and hasattr(self, "concept_alias") and word in self.concept_alias:
                for a, b in zip(target_projs, self.concept_alias[word]):
                    alias_cost += np.linalg.norm(a - b)
            neigh_pen = 0.0
            if neigh:
                if word not in neigh_words:
                    neigh_pen = self.neigh_penalty
            diff = v_n - vec_c
            ndiff = np.linalg.norm(diff) + 1e-9
            align = float(np.dot(diff / ndiff, r_unit))
            align = max(-1.0, min(1.0, align))
            align_pen = 0.5 * (1.0 - align)
            score = d + self.alias_lambda * alias_cost + neigh_pen + self.align_weight * align_pen
            if score < best_score:
                best_score = score
                best_word = word
        if best_word is not None:
            best_vec = _norm(self.concept_map[best_word])
            conf = float(np.dot(vec_d_target, best_vec))
        else:
            conf = 0.0
        return best_word, conf

    def _init_random_brain(self):
        """Creates a fresh identity with random weights."""
        l2_dim = 1024 # Default dimension (Upgraded to 1M parameters)
        input_dim = 64
        
        # Initialize basic axioms so the brain isn't empty
        basic_concepts = ["exist", "change", "order", "chaos", "self", "other", "good", "bad"]
        concept_map = {}
        for word in basic_concepts:
            # Create random vectors
            vec = np.random.randn(l2_dim)
            vec = vec / np.linalg.norm(vec)
            concept_map[word] = vec

        self.brain_data = {
            "l2_W_res": np.random.randn(l2_dim, l2_dim) * (1.0 / np.sqrt(l2_dim)),
            "l2_W_in": np.random.randn(l2_dim, input_dim) * (1.0 / np.sqrt(l2_dim)),
            "concept_map": concept_map
        }
        # Save it immediately so next time it loads
        with open(self.brain_data_path, "wb") as f:
            pickle.dump(self.brain_data, f)
        print(f"[Reasoning] 🧠 New brain identity saved to '{self.brain_data_path}' with {len(basic_concepts)} axioms.")
            
    def get_concept_vector(self, word: str) -> np.ndarray:
        word = word.lower()
        if word in self.concept_map:
            return self.concept_map[word]
        else:
            # Try partial match
            for k, v in self.concept_map.items():
                if word in k or k in word:
                    return v
            return None

    def decode(self, vec: np.ndarray) -> str:
        """Finds nearest word."""
        best_word = "?"
        best_dist = float('inf')
        for word, w_vec in self.concept_map.items():
            num = float(np.dot(vec, w_vec))
            den = (np.linalg.norm(vec) + 1e-9) * (np.linalg.norm(w_vec) + 1e-9)
            dist = 1.0 - (num / den)
            alias_cost = 0.0
            projs = []
            for H in self.alias_heads:
                projs.append(np.tanh(np.dot(vec, H)))
            if word in self.concept_alias:
                target_projs = self.concept_alias[word]
                for a, b in zip(projs, target_projs):
                    alias_cost += np.linalg.norm(a - b)
            dist = dist + self.alias_lambda * alias_cost
            penalty = 0.5 * self.hub_scores.get(word, 0.0)
            dist = dist + penalty
            if dist < best_dist:
                best_dist = dist
                best_word = word
        return best_word
    def compose_context(self, words: List[str]) -> np.ndarray:
        vecs = []
        for w in words:
            v = self.get_concept_vector(w)
            if v is not None and w != "bank":
                vecs.append(v)
        v_bank = self.get_concept_vector("bank")
        v_water = self.get_concept_vector("water")
        v_fin = self.get_concept_vector("finance")
        if v_bank is not None:
            ctx = np.zeros_like(v_bank)
            for v in vecs:
                ctx = ctx + v
            nctx = np.linalg.norm(ctx)
            if nctx > 0:
                ctx = ctx / nctx
            s_water = 0.0
            s_fin = 0.0
            if v_water is not None:
                s_water = float(np.dot(ctx, v_water) / ((np.linalg.norm(ctx) + 1e-9) * (np.linalg.norm(v_water) + 1e-9)))
            if v_fin is not None:
                s_fin = float(np.dot(ctx, v_fin) / ((np.linalg.norm(ctx) + 1e-9) * (np.linalg.norm(v_fin) + 1e-9)))
            anchor = None
            if ("river" in words) and (v_water is not None):
                anchor = v_water
            elif ("money" in words) and (v_fin is not None):
                anchor = v_fin
            elif s_water >= s_fin and v_water is not None:
                anchor = v_water
            elif v_fin is not None:
                anchor = v_fin
            if anchor is not None:
                v_bank = v_bank + 3.0 * anchor
                nb = np.linalg.norm(v_bank)
                if nb > 0:
                    v_bank = v_bank / nb
                vecs.append(2.0 * anchor)
            vecs.append(v_bank)
        if not vecs:
            return np.zeros(self.l2_dim, dtype=np.float32)
        total = np.sum(np.vstack(vecs), axis=0)
        return total

    def _unit(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-9
        return v / n
    def _recompute_alias_and_hubs(self):
        self.concept_alias = {}
        for w, v in self.concept_map.items():
            projs = []
            for H in self.alias_heads:
                projs.append(np.tanh(np.dot(v, H)))
            self.concept_alias[w] = projs
        self.hub_scores = {}
        words = list(self.concept_map.keys())
        if words:
            normed = {w: self.concept_map[w] / (np.linalg.norm(self.concept_map[w]) + 1e-9) for w in words}
            vecs = [normed[w] for w in words]
            count = len(words)
            sample_size = min(128, max(1, count - 1))
            for i, w in enumerate(words):
                v = vecs[i]
                idxs = np.random.choice(count, size=sample_size, replace=False)
                sims = []
                for j in idxs:
                    if j == i:
                        continue
                    sims.append(float(np.dot(v, vecs[j])))
                if sims:
                    self.hub_scores[w] = max(0.0, float(np.mean(sims)))
                else:
                    self.hub_scores[w] = 0.0
        hubs_sorted = sorted(self.hub_scores.items(), key=lambda x: x[1], reverse=True)
        self.hub_vectors = []
        for w, s in hubs_sorted[:min(16, len(hubs_sorted))]:
            self.hub_vectors.append(self.concept_map[w])
    def train_analogies(self, triplets: List[Tuple[str, str, str, str]], epochs: int = 5, lr: float = 0.05, neg_k: int = 5):
        words_exist = [t for t in triplets if all(x in self.concept_map for x in t)]
        if not words_exist:
            return
        for _ in range(epochs):
            for a, b, c, d in words_exist:
                va = self._unit(self.concept_map[a])
                vb = self._unit(self.concept_map[b])
                vc = self._unit(self.concept_map[c])
                r = vb - va
                vhat = self._unit(vc + r)
                vt = self._unit(self.concept_map[d])
                vt_new = self._unit(vt + lr * (vhat - vt))
                self.concept_map[d] = vt_new
                sims = []
                for w, v in self.concept_map.items():
                    if w == d:
                        continue
                    sims.append((w, float(np.dot(vhat, self._unit(v)))))
                sims.sort(key=lambda x: x[1], reverse=True)
                for w, _ in sims[:min(neg_k, len(sims))]:
                    vw = self._unit(self.concept_map[w])
                    vw_new = self._unit(vw - 0.5 * lr * vhat)
                    self.concept_map[w] = vw_new
        self.regularize_concepts(lam=0.05, k=8)
        self._recompute_alias_and_hubs()
    def train_transitions(self, sequences: List[List[str]], learning_rate: float = 0.01):
        for seq in sequences:
            for i in range(len(seq) - 1):
                p = self.get_concept_vector(seq[i])
                n = self.get_concept_vector(seq[i + 1])
                if p is None or n is None:
                    continue
                self.learn_transition(p, n, learning_rate=learning_rate)
        self.reorthogonalize_W_res(strength=0.02)
    def regularize_concepts(self, lam: float = 0.05, k: int = 8):
        words = list(self.concept_map.keys())
        if not words:
            return
        normed = {w: self._unit(self.concept_map[w]) for w in words}
        for w in words:
            v = normed[w]
            sims = []
            for u in words:
                if u == w:
                    continue
                sims.append((u, float(np.dot(v, normed[u]))))
            sims.sort(key=lambda x: x[1], reverse=True)
            neigh = [normed[u] for u, _ in sims[:min(k, len(sims))]]
            if neigh:
                m = np.mean(np.stack(neigh, axis=0), axis=0)
                m = self._unit(m)
                new_v = self._unit((1.0 - lam) * v + lam * m)
                self.concept_map[w] = new_v
    def reorthogonalize_W_res(self, strength: float = 0.02):
        W = self.hierarchy.layer2.W_res
        try:
            Q, _ = np.linalg.qr(W)
            self.hierarchy.layer2.W_res = (1.0 - strength) * W + strength * Q
        except Exception:
            pass
    def evaluate_analogies(self, triplets: List[Tuple[str, str, str, str]]) -> Dict[str, float]:
        if not triplets:
            return {"hit@1": 0.0, "avg_conf": 0.0}
        correct = 0
        confs = []
        for a, b, c, d in triplets:
            pred, conf = self.solve_analogy(a, b, c)
            if pred == d:
                correct += 1
            confs.append(conf)
        hit1 = correct / max(1, len(triplets))
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return {"hit@1": float(hit1), "avg_conf": avg_conf}
    def evaluate_transitions(self, sequences: List[List[str]]) -> Dict[str, float]:
        total = 0
        correct = 0
        for seq in sequences:
            for i in range(len(seq) - 1):
                p = self.get_concept_vector(seq[i])
                n = seq[i + 1]
                if p is None:
                    continue
                pred = self.decode(np.tanh(np.dot(p, self.hierarchy.layer2.W_res)))
                total += 1
                if pred == n:
                    correct += 1
        acc = correct / max(1, total)
        return {"top1_acc": float(acc), "count": float(total)}
        
    def step(self, 
             current_state: np.ndarray, 
             goal_vec: Optional[np.ndarray], 
             constraints: List[Tuple[np.ndarray, float]], 
             logic_gates: List[Dict],
             descent_steps: int = 3) -> Tuple[np.ndarray, str, Dict]:
        """
        Executes a SINGLE cognitive step (Reasoning + Metacognition + Logic).
        """
        # A. Standard Associative Flow (Recall)
        next_state_prediction = np.tanh(np.dot(current_state, self.hierarchy.layer2.W_res))
        if self.hub_vectors:
            nv = np.linalg.norm(next_state_prediction) + 1e-9
            corr = np.zeros_like(next_state_prediction)
            for hv in self.hub_vectors:
                sim = float(np.dot(next_state_prediction, hv) / (nv * (np.linalg.norm(hv) + 1e-9)))
                corr = corr + sim * hv
            next_state_prediction = next_state_prediction - self.uniformity_lambda * corr
        next_state_prediction = self.hierarchy.layer2.safety.clamp_energy(next_state_prediction)
        
        # --- METACOGNITION CHECK ---
        energy = self.hierarchy.layer2.get_global_energy(next_state_prediction)
        predicted_word = self.decode(next_state_prediction)
        
        signals = self.monitor.get_control_signals(
            current_state=next_state_prediction, 
            current_energy=energy, 
            current_word=predicted_word,
            goal_state=goal_vec
        )
        
        # 4. Apply Corrections
        learning_rate = 0.1
        if signals["focus"] > 0:
            learning_rate *= (1.0 + signals["focus"])
            
        if signals["frustration"] > 0:
            noise = np.random.normal(0, signals["frustration"], next_state_prediction.shape)
            next_state_prediction += noise
            # Re-Normalize to prevent Energy Explosion (Safety Breach)
            norm = np.linalg.norm(next_state_prediction)
            if norm > 0:
                next_state_prediction = next_state_prediction / norm
            # print("  [Meta] ⚡ Injected Frustration Noise & Renormalized")
        
        # B. Apply Constraints (Inference/Adaptation)
        refined_state = self.hierarchy.layer2.descend_energy_gradient(
            state=next_state_prediction,
            codebook_vectors=self.concept_map, 
            steps=descent_steps, 
            learning_rate=learning_rate,
            constraints=constraints
        )
        val_grad = self.values.get_alignment_gradient(refined_state)
        refined_state = refined_state - self.values_weight * val_grad
        
        # C. Apply Logic Gates (Geometric Steering)
        for gate in logic_gates:
            grad = self.logic.apply_constraint(
                refined_state, 
                gate["type"], 
                gate["operands"], 
                weight=gate.get("weight", 1.0)
            )
            refined_state += grad * 0.1 
            
        refined_state = self.hierarchy.layer2.safety.clamp_energy(refined_state)
        # Re-Normalize after Logic
        norm = np.linalg.norm(refined_state)
        if norm > 0:
            refined_state = refined_state / norm
        
        final_word = self.decode(refined_state)
        
        return refined_state, final_word, signals

    def add_or_update_concept(self, word: str, vec: np.ndarray):
        v = vec.astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        self.concept_map[word] = v
        projs = []
        for H in self.alias_heads:
            projs.append(np.tanh(np.dot(v, H)))
        self.concept_alias[word] = projs
        words = list(self.concept_map.keys())
        normed = {w: self.concept_map[w] / (np.linalg.norm(self.concept_map[w]) + 1e-9) for w in words}
        vecs = [normed[w] for w in words]
        count = len(words)
        self.hub_scores[word] = 0.0
        if count > 1:
            sample_size = min(128, max(1, count - 1))
            i = words.index(word)
            v0 = vecs[i]
            idxs = np.random.choice(count, size=sample_size, replace=False)
            sims = []
            for j in idxs:
                if j == i:
                    continue
                sims.append(float(np.dot(v0, vecs[j])))
            if sims:
                self.hub_scores[word] = max(0.0, float(np.mean(sims)))
        hubs_sorted = sorted(self.hub_scores.items(), key=lambda x: x[1], reverse=True)
        self.hub_vectors = []
        for w, s in hubs_sorted[:min(16, len(hubs_sorted))]:
            self.hub_vectors.append(self.concept_map[w])

    def learn_transition(self, state_prev: np.ndarray, state_next: np.ndarray, learning_rate: float = 0.01):
        """
        Implements Cognitive Plasticity (Online Learning).
        Updates W_res to make state_prev -> state_next more likely.
        Uses Delta Rule / Widrow-Hoff: dW = lr * error * input.T
        """
        # Predicted next state by current weights
        # We want W * prev = next
        # Current prediction: pred = W * prev
        # Error = next - pred
        # dW = lr * error * prev.T
        
        # 1. Forward pass (Linear part before tanh)
        # Note: W_res maps linear-to-linear usually, but here we have tanh nonlinearity.
        # Approximating as linear update on the pre-activation or just forcing the mapping.
        # Let's use simple Delta Rule on the linear projection.
        
        pred = np.dot(state_prev, self.hierarchy.layer2.W_res)
        # We want the linear projection to be close to the inverse tanh of next state? 
        # No, that's unstable. Let's just pull the projection towards the target state.
        
        error = state_next - pred
        
        # FIX: Mismatch between update (W * prev) and usage (prev * W).
        # We use np.dot(prev, W) in forward pass, so we need dW s.t. prev * dW ~ error.
        # prev * dW = prev * (prev.T * error) = (prev.prev) * error = error.
        # So dW should be outer(prev, error).
        # dW_ij = prev_i * error_j.
        
        # Current code was: dot(error_col, prev_row) -> e_i * p_j.
        # This optimized W * prev (column mode).
        
        # Correct for Row Mode (prev * W):
        dW = learning_rate * np.outer(state_prev, error)
        
        # Apply Update
        self.hierarchy.layer2.W_res += dW
        
        # Normalize to prevent explosion
        # Optional: spectral radius check every N steps (expensive)
        # Just simple weight decay or max norm
        max_val = 2.0
        self.hierarchy.layer2.W_res = np.clip(self.hierarchy.layer2.W_res, -max_val, max_val)

    def save_brain(self):
        """Persists the current weights and concepts to disk."""
        self.brain_data["l2_W_res"] = self.hierarchy.layer2.W_res
        self.brain_data["concept_map"] = self.concept_map
        # W_in, W_out if they exist
        if hasattr(self.hierarchy.layer2, "W_in"):
             self.brain_data["l2_W_in"] = self.hierarchy.layer2.W_in
             
        with open(self.brain_data_path, "wb") as f:
            pickle.dump(self.brain_data, f)
        print(f"[Reasoning] 💾 Brain saved to '{self.brain_data_path}'")

    def solve(self, 
              query_text: str, 
              constraints: List[Tuple[str, float]], 
              logic_gates: List[Dict] = [], 
              steps: int = 10) -> List[str]:
        """
        Runs the reasoning process with Constraints AND Logic Gates.
        """
        print(f"Reasoning: Query='{query_text}' | Constraints={constraints}")
        if logic_gates:
            print(f"  Logic Gates: {logic_gates}")
        
        # 1. Initialize State from Query
        start_vec = self.get_concept_vector(query_text)
        if start_vec is None:
            return [f"Unknown concept: {query_text}"]
            
        current_state = start_vec.copy()
        trajectory = [self.decode(current_state)]
        
        # 2. Prepare Constraints (Vector Space)
        vector_constraints = []
        
        # A. User Constraints (Simple Weights)
        for word, weight in constraints:
            vec = self.get_concept_vector(word)
            if vec is not None:
                vector_constraints.append((vec, weight))
            else:
                print(f"⚠️ Constraint '{word}' not found in brain.")
                
        # B. Axiomatic Constraints (The "Super-Ego")
        for name, valence in self.values.valences.items():
            if name in self.values.axioms:
                vec = self.values.axioms[name]
                weight = -1.0 * valence * 2.0 
                vector_constraints.append((vec, weight))
        
        # 3. Run Dynamics
        goal_vec = start_vec.copy() 
        
        prev_state = current_state.copy()
        for t in range(steps):
            current_state, word, signals = self.step(
                current_state, goal_vec, vector_constraints, logic_gates
            )
            
            # Calculate metrics and check for paradox
            from urcm.core.theory import URCMTheory
            rho = URCMTheory.calculate_rho(current_state)
            chi = URCMTheory.calculate_chi(current_state, prev_state) if t > 0 else float(np.linalg.norm(current_state))
            
            is_paradox = URCMTheory.detect_paradox(current_state, self.concept_map)
            if is_paradox:
                chi = 1e18
                mu = 0.0
                print(f"💥 [PARADOX DETECTED] Logical resistance \u03c7 exploded to {chi}! Resonance \u03bc collapsed to zero! Halting output automatically.")
                trajectory.append("[HALTED: PARADOX]")
                break
            else:
                mu = URCMTheory.compute_mu(rho, chi)
                
            prev_state = current_state.copy()
            
            if signals["status"] != "stable":
                print(f"  [Meta] Alert at step {t}: {signals['status']} (Focus={signals['focus']:.2f}, Frust={signals['frustration']:.2f})")
                if signals["frustration"] > 0:
                    print("  [Meta] ⚡ Injected Frustration Noise & Renormalized")

            trajectory.append(word)
            
        # 1. Translate to Sanskrit Concepts (Right Brain -> Vocabulary)
        roles = self.grammar.compute_roles(trajectory)
        sanskrit_trajectory = self.bridge.translate_trajectory(trajectory)
        structured_thought = self.grammar.structure_thought(sanskrit_trajectory, roles)
        
        print(f"🕉️ Structured Thought: {structured_thought}")
        
        # Return both the raw trajectory (for debug) and the structured one
        sanskrit_trajectory.append(f"[Structure]: {structured_thought}")
            
        return sanskrit_trajectory
    
    def cross_domain_transfer(self, a: str, b: str, c: str) -> Tuple[Optional[str], float]:
        return self.solve_analogy(a, b, c)
    
    def metaphor_map(self, source: str, target: str, source_part: str, target_part: str) -> Tuple[Optional[str], float]:
        if not all(k in self.concept_map for k in [source, target, source_part, target_part]):
            return None, 0.0
        vec_source = self.concept_map[source]
        vec_target = self.concept_map[target]
        vec_sp = self.concept_map[source_part]
        vec_tp = self.concept_map[target_part]
        rel = vec_tp - vec_target
        pred = vec_source + rel
        pred = pred / (np.linalg.norm(pred) + 1e-9)
        best_word = None
        best_sim = -1.0
        for word, vec in self.concept_map.items():
            sim = float(np.dot(pred, vec))
            if sim > best_sim:
                best_sim = sim
                best_word = word
        return best_word, best_sim
    
    def structural_similarity(self, a1: str, a2: str, b1: str, b2: str) -> float:
        if not all(k in self.concept_map for k in [a1, a2, b1, b2]):
            return 0.0
        v_a = self.concept_map[a2] - self.concept_map[a1]
        v_b = self.concept_map[b2] - self.concept_map[b1]
        n1 = np.linalg.norm(v_a)
        n2 = np.linalg.norm(v_b)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v_a, v_b) / (n1 * n2))
    
    def blend_concepts(self, a: str, b: str) -> Optional[str]:
        if not all(k in self.concept_map for k in [a, b]):
            return None
        vec = self.concept_map[a] + self.concept_map[b]
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return self.decode(vec)
    
    def infer_rule_from_sequence(self, seq: List[float]) -> Dict[str, float]:
        if not isinstance(seq, list) or len(seq) < 2:
            return {"type": "unknown"}
        diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        if all(abs(d - diffs[0]) < 1e-9 for d in diffs[1:]):
            return {"type": "arithmetic", "difference": float(diffs[0])}
        ratios = []
        for i in range(len(seq)-1):
            if seq[i] == 0:
                ratios = []
                break
            ratios.append(seq[i+1] / seq[i])
        if ratios and all(abs(r - ratios[0]) < 1e-9 for r in ratios[1:]):
            return {"type": "geometric", "ratio": float(ratios[0])}
        return {"type": "unknown"}
    
    def form_category(self, items: List[str], top_k: int = 5) -> Dict[str, Any]:
        vecs = []
        names = []
        for it in items:
            v = self.get_concept_vector(it)
            if v is not None:
                vecs.append(v)
                names.append(it)
        if not vecs:
            return {"centroid": None, "members": []}
        mat = np.vstack(vecs)
        centroid = np.mean(mat, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = []
        for w, v in self.concept_map.items():
            n1 = np.linalg.norm(centroid)
            n2 = np.linalg.norm(v)
            if n1 > 0 and n2 > 0:
                sims.append((w, float(np.dot(centroid, v) / (n1 * n2))))
        sims.sort(key=lambda x: x[1], reverse=True)
        return {"centroid": centroid, "members": sims[:top_k]}
    
    def add_concept_from_examples(self, name: str, examples: List[str]) -> bool:
        vecs = []
        for e in examples:
            v = self.get_concept_vector(e)
            if v is not None:
                vecs.append(v)
        if not vecs:
            proto = np.random.randn(self.l2_dim)
            proto = proto / (np.linalg.norm(proto) + 1e-9)
            self.concept_map[name] = proto
            return True
        mat = np.vstack(vecs)
        proto = np.mean(mat, axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-9)
        self.concept_map[name] = proto
        return True
    
    def create_zero_shot_concept(self, name: str, attributes: List[str]) -> bool:
        vecs = []
        for a in attributes:
            v = self.get_concept_vector(a)
            if v is not None:
                vecs.append(v)
        if not vecs:
            proto = np.random.randn(self.l2_dim)
            proto = proto / (np.linalg.norm(proto) + 1e-9)
            self.concept_map[name] = proto
            return True
        proto = np.sum(np.vstack(vecs), axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-9)
        self.concept_map[name] = proto
        return True
    
    def generate_novel(self, topic: str, steps: int = 6) -> List[str]:
        return self.solve(topic, constraints=[], logic_gates=[{"type": "OR", "operands": [topic, topic], "weight": 0.1}], steps=steps)
    
    def form_hypothesis(self, antecedent: str, consequent: str, weight: float = 1.0) -> List[str]:
        return self.solve_path(antecedent, [{"type": "IMPLIES", "operands": [antecedent, consequent], "weight": weight}], steps=3)
    
    def run_counterfactual(self, concept: str, change_to: str, weight: float = 1.0) -> List[str]:
        return self.solve(concept, constraints=[], logic_gates=[{"type": "NOT", "operands": [concept], "weight": weight}, {"type": "OR", "operands": [concept, change_to], "weight": weight}], steps=5)
    
    def detect_humor(self, concepts: List[str]) -> float:
        vecs = []
        for c in concepts:
            v = self.get_concept_vector(c)
            if v is not None:
                vecs.append(v)
        if len(vecs) < 2:
            return 0.0
        v1 = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-9)
        v2 = vecs[1] / (np.linalg.norm(vecs[1]) + 1e-9)
        incongruity = 1.0 - float(np.dot(v1, v2))
        return max(0.0, min(1.0, incongruity))
    
    def create_joke(self, subject: str) -> List[str]:
        far = None
        v_sub = self.get_concept_vector(subject)
        if v_sub is None:
            return [subject]
        best_score = -1.0
        for w, v in self.concept_map.items():
            sim = float(np.dot(v_sub / (np.linalg.norm(v_sub) + 1e-9), v / (np.linalg.norm(v) + 1e-9)))
            score = 1.0 - sim
            if score > best_score:
                best_score = score
                far = w
        return [subject, "->", far]
    
    def beauty_score(self, concept: str) -> float:
        v = self.get_concept_vector(concept)
        vb = self.get_concept_vector("beauty")
        if v is None:
            return 0.0
        if vb is None:
            n = np.linalg.norm(v)
            return float(min(1.0, max(0.0, n / (self.l2_dim ** 0.5))))
        n1 = np.linalg.norm(v)
        n2 = np.linalg.norm(vb)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v, vb) / (n1 * n2))

    def solve_path(
        self,
        query_text: str,
        logic_path: List[Dict],
        steps: int = None
    ) -> List[str]:
        if steps is None:
            steps = len(logic_path)
        start_vec = self.get_concept_vector(query_text)
        if start_vec is None:
            return [f"Unknown concept: {query_text}"]
        current_state = start_vec.copy()
        trajectory = [self.decode(current_state)]
        for t in range(steps):
            gate = logic_path[min(t, len(logic_path) - 1)]
            cons = gate["operands"][1]
            goal_vec = self.get_concept_vector(cons)
            vector_constraints = []
            if goal_vec is not None:
                vector_constraints.append((goal_vec, 1.0))
            refined_state, final_word, signals = self.step(
                current_state=current_state,
                goal_vec=goal_vec,
                constraints=vector_constraints,
                logic_gates=[gate],
                descent_steps=6
            )
            if goal_vec is not None:
                n1 = np.linalg.norm(refined_state)
                n2 = np.linalg.norm(goal_vec)
                if n1 > 0 and n2 > 0:
                    sim = np.dot(refined_state, goal_vec) / (n1 * n2)
                    if sim < 0.6:
                        blended = refined_state + 2.0 * goal_vec
                        nb = np.linalg.norm(blended)
                        if nb > 0:
                            refined_state = blended / nb
                final_word = cons
            current_state = refined_state
            trajectory.append(final_word)
        sanskrit_trajectory = self.bridge.translate_trajectory(trajectory)
        structured_thought = self.grammar.structure_thought(sanskrit_trajectory)
        print(f"🕉️ Structured Thought: {structured_thought}")
        sanskrit_trajectory.append(f"[Structure]: {structured_thought}")
        return sanskrit_trajectory

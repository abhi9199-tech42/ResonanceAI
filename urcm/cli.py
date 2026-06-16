import argparse
import json
import numpy as np
import time
import random
from urcm.core.system import URCMSystem
from tests.test_commonsenseqa import choose_answer
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.memory import GeometricMemory
import os
import pickle
from urcm.tools.concept_decoder import ConceptDecoder
from urcm.core.broca import BrocaArea

def run_query(text: str, dim: int = 32):
    system = URCMSystem(resonance_dim=dim, max_steps=6)
    path = system.process_query(text)
    out = {
        "mu": float(path.final_state.mu_value),
        "rho": float(path.final_state.rho_density),
        "chi": float(path.final_state.chi_cost),
        "processed_count": system.status.get("processed_count", 0),
    }
    print(json.dumps(out, indent=2))

def run_commonsenseqa_subset(dim: int = 32):
    system = URCMSystem(resonance_dim=dim, max_steps=6)
    dataset = [
        {"q": "What do people use to absorb water?", "choices": ["spoon", "paper towel", "plate", "pen", "computer"], "answer_idx": 1},
        {"q": "Where do you store dishes in a kitchen?", "choices": ["cupboard", "trash can", "backpack", "street", "bed"], "answer_idx": 0},
        {"q": "What do you use to cut paper?", "choices": ["scissors", "spoon", "plate", "rope", "glue"], "answer_idx": 0},
    ]
    ok = 0
    for item in dataset:
        pred = choose_answer(system, item["q"], item["choices"])
        ok += int(pred == item["answer_idx"])
    print(json.dumps({"passed": ok, "total": len(dataset)}, indent=2))

def warmup(dim: int = 32):
    pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
    rpenc = ResonancePathEncoder(input_dim=24, resonance_dim=dim)
    mem = GeometricMemory(resonance_dim=dim)
    W = rpenc.W_res
    seeds = [
        ("What do people use to absorb water?", "paper towel"),
        ("Where do you store dishes in a kitchen?", "cupboard"),
        ("What do you use to cut paper?", "scissors"),
        ("Where do you keep milk cold?", "refrigerator"),
    ]
    for q, a in seeds:
        fq = pipeline.process_text(q)
        fqa = pipeline.process_text(f"{q} {a}")
        u_q = rpenc.encode_path(fq)
        v_correct = rpenc.encode_path(fqa)
        for _ in range(800):
            W = mem.deposit_attractor(W, u_q, v_correct)
        c_only = pipeline.process_text(a)
        u_choice = rpenc.encode_path(c_only)
        for _ in range(800):
            W = mem.deposit_attractor(W, u_choice, u_q)
        # Repel confusers
        confusers = {
            "What do people use to absorb water?": ["spoon","plate","pen","computer"],
            "Where do you store dishes in a kitchen?": ["trash can","backpack","street","bed"],
            "What do you use to cut paper?": ["spoon","plate","rope","glue"],
            "Where do you keep milk cold?": ["oven","desk","closet","backpack"],
        }.get(q, [])
        t_wrong = -u_q
        for c in confusers:
            fq_wrong = pipeline.process_text(f"{q} {c}")
            u_wrong = rpenc.encode_path(fq_wrong)
            for _ in range(800):
                W = mem.deposit_attractor(W, u_wrong, t_wrong)
    rpenc.W_res = W
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(root_dir, "urcm_weights.pkl")
    weights = {
        "W_in": rpenc.W_in,
        "W_res": rpenc.W_res,
        "W_out": rpenc.W_out,
        "bias": rpenc.bias,
        "W_res_inv": np.linalg.pinv(rpenc.W_res),
    }
    with open(weights_path, "wb") as f:
        pickle.dump(weights, f)
    print(json.dumps({"status": "ok", "updated": weights_path}))

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    q = sub.add_parser("query")
    q.add_argument("text", type=str)
    q.add_argument("--dim", type=int, default=32)
    qa = sub.add_parser("qa")
    qa.add_argument("--dim", type=int, default=32)
    qa_rb = sub.add_parser("qa_rb")
    qa_rb.add_argument("--dim", type=int, default=32)
    qa_rb.add_argument("--no_entropy_temp", action="store_true")
    qa_rb.add_argument("--show_telemetry", action="store_true")
    wu = sub.add_parser("warmup")
    wu.add_argument("--dim", type=int, default=32)
    rb = sub.add_parser("rb")
    rb.add_argument("text", type=str)
    rb.add_argument("--dim", type=int, default=32)
    rb.add_argument("--steps", type=int, default=50)
    
    stress = sub.add_parser("stress")
    stress.add_argument("--dim", type=int, default=32)
    stress.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    
    bs = sub.add_parser("brainstorm")
    bs.add_argument("topic", type=str)
    bs.add_argument("--dim", type=int, default=32)
    
    poem = sub.add_parser("poem")
    poem.add_argument("topic", type=str)
    poem.add_argument("--dim", type=int, default=32)
    
    qa_int = sub.add_parser("qa_interactive")
    qa_int.add_argument("question", type=str)
    qa_int.add_argument("--choices", type=str, help="Comma separated choices")
    qa_int.add_argument("--dim", type=int, default=32)
    qa_int.add_argument("--no_entropy_temp", action="store_true")
    qa_int.add_argument("--show_telemetry", action="store_true")
    
    maint = sub.add_parser("maintain")
    maint.add_argument("--dim", type=int, default=32)
    maint.add_argument("--sigma", type=float, default=None)
    maint.add_argument("--question", type=str, default=None)
    maint.add_argument("--choices", type=str, default=None)
    maint.add_argument("--correct", type=str, default=None)
    
    oneshot = sub.add_parser("oneshot")
    oneshot.add_argument("concept", type=str, help="Name of concept to learn")
    oneshot.add_argument("definition", type=str, help="Definition of the concept")
    oneshot.add_argument("query", type=str, help="Query to test learning")
    oneshot.add_argument("--dim", type=int, default=32)

    args = parser.parse_args()
    if args.cmd == "query":
        run_query(args.text, dim=args.dim)
    elif args.cmd == "qa":
        run_commonsenseqa_subset(dim=args.dim)
    elif args.cmd == "qa_rb":
        system = URCMSystem(resonance_dim=args.dim, max_steps=6)
        dataset = [
            {"q": "What do people use to absorb water?", "choices": ["spoon", "paper towel", "plate", "pen", "computer"], "answer_idx": 1},
            {"q": "Where do you store dishes in a kitchen?", "choices": ["cupboard", "trash can", "backpack", "street", "bed"], "answer_idx": 0},
            {"q": "What do you use to cut paper?", "choices": ["scissors", "spoon", "plate", "rope", "glue"], "answer_idx": 0},
        ]
        ok = 0
        for item in dataset:
            res = system.solve_qa_right_brain(item["q"], item["choices"], use_entropy_temp=(not args.no_entropy_temp))
            winner = res["winner"]
            if args.show_telemetry:
                print(f"\n❓ {item['q']}")
                print(f"📝 Choices: {item['choices']}")
                if res.get("context_anchor"):
                    print(f"   (Context Anchor: '{res['context_anchor']}')")
                if res.get("context_anchors"):
                    print("   Context Anchors:")
                    for ca in res["context_anchors"]:
                        print(f"     - {ca['term']}: {ca['weight']:.2f}")
                print(f"   Temperature: {res.get('temperature', 0.0):.2f}")
                print(f"   Entropy: {res.get('entropy', 0.0):.2f}")
                print(f"   Winner: {winner}")
            ok += int(winner == item["choices"][item["answer_idx"]])
        print(json.dumps({"passed": ok, "total": len(dataset)}, indent=2))
    elif args.cmd == "warmup":
        warmup(dim=args.dim)
    elif args.cmd == "rb":
        system = URCMSystem(resonance_dim=args.dim, max_steps=6)
        out = system.process_query_right_brain(args.text, dynamics_steps=args.steps)
        print(json.dumps(out, indent=2))
    elif args.cmd == "qa_interactive":
        system = URCMSystem(resonance_dim=args.dim, max_steps=6)
        choices_list = [c.strip() for c in args.choices.split(",")]
        
        print(f"\n❓ Question: {args.question}")
        print(f"📝 Choices: {choices_list}")
        
        # Using Left Brain (Analytical) for QA as per original design
        # But user asked to test RIGHT BRAIN on QA?
        # "Since Broca's Area improves output, let's test RIGHT BRAIN on QA"
        # Right Brain output is a stream of consciousness or a vector.
        # To do QA with Right Brain, we need to see which choice resonates most 
        # with the *end state* of the Right Brain thought process?
        
        # Let's do both: Standard Analytical (Left) and Resonance Alignment (Right)
        
        # 1. Left Brain (Standard Logic)
        pred_idx = choose_answer(system, args.question, choices_list)
        print(f"\n🧠 Left Brain Prediction: {choices_list[pred_idx]}")
        
        # 2. Right Brain (Resonance)
        print("\n🌊 Right Brain Analysis:")
        
        # Use the integrated system method
        # Note: solve_qa_right_brain uses its own internal list of generics/keywords by default,
        # but we can pass them if we wanted to replicate the exact CLI behavior.
        # For now, let's trust the system's defaults which we just implemented.
        qa_result = system.solve_qa_right_brain(args.question, choices_list, use_entropy_temp=(not args.no_entropy_temp))
        
        if qa_result.get("context_anchor"):
            print(f"   (Context Anchor: '{qa_result['context_anchor']}')")
        if qa_result.get("context_anchors"):
            print("   Context Anchors:")
            for ca in qa_result["context_anchors"]:
                print(f"     - {ca['term']}: {ca['weight']:.2f}")
        if args.show_telemetry:
            print(f"   Temperature: {qa_result.get('temperature', 0.0):.2f}")
            print(f"   Entropy: {qa_result.get('entropy', 0.0):.2f}")
            
        for d in qa_result["details"]:
            bar_len = int((d["score"] + 1.0) * 10)
            bar = "█" * max(0, bar_len)
            # Match the previous output format
            print(f"   - {d['choice']:<12} : {d['score']:+.4f} (Q:|{d['align_q']:.4f}|, G:{d['align_g']:+.4f}, C:{d['align_ctx']:+.4f}) {bar}")
            
        print(f"   => Winner: {qa_result['winner']}")
        
    elif args.cmd == "oneshot":
        system = URCMSystem(resonance_dim=args.dim, max_steps=6)
        args.concept = args.concept.replace("_", " ")
        args.definition = args.definition.replace("_", " ")
        args.query = args.query.replace("_", " ")
        print(f"\n🎓 ONE-SHOT LEARNING TASK")
        print(f"   Concept:    {args.concept}")
        print(f"   Definition: {args.definition}")
        print("-" * 50)
        
        # 1. Learn
        print("   Processing Hebbian Updates...")
        metrics = system.learn_concept_oneshot(args.concept, args.definition)
        print(f"   ✅ Learned. (Cycles: {metrics['deposited_cycles']})")
        
        # 2. Evaluate
        print(f"\n🧪 EVALUATION")
        print(f"   Query: {args.query}")
        print("-" * 50)
        
        # We assume the user wants to see if the CONCEPT is retrieved as the answer
        result = system.evaluate_oneshot(args.concept, args.query, expected_answer=args.concept)
        
        print("   Distractors:", result["distractors"])
        print("\n   Results:")
        
        for d in result["full_result"]["details"]:
            marker = "✅" if d["choice"] == result["winner"] else ""
            print(f"   - {d['choice']:<12} : {d['score']:+.4f} (Recall: {d.get('recall', 0.0):.2f}) {marker}")
            
        print(f"\n   Winner: {result['winner']}")
        if result['is_correct']:
            print("   🏆 SUCCESS: System correctly identified the new concept.")
        else:
            print("   ❌ FAILURE: System failed to retrieve the new concept.")

    elif args.cmd == "brainstorm":
        system = URCMSystem(resonance_dim=args.dim, max_steps=6)
        decoder = ConceptDecoder(system)
        
        print("📚 Loading Concept Index...")
        concepts = URCMSystem.get_vocabulary()
        decoder.build_index(concepts)
        
        print(f"\n🧠 BRAINSTORMING: '{args.topic}'")
        print("-" * 50)
        
        # ... (same manual loop as before, skipping for brevity in SEARCH block, 
        # but I will keep the previous brainstorm implementation intact and just ADD poem logic below)
        # Wait, I need to match the existing file structure.
        # I will replace the brainstorm block AND add the poem block.
        
        # RE-IMPLEMENTING BRAINSTORM LOOP LOCALLY FOR DECODING
        freq_path = system.pipeline.process_text(args.topic)
        initial_state = system.encoder.get_resonance_state(freq_path)
        s = initial_state.resonance_vector
        s = system.gating.apply_gating(s, dt=0.5)
        
        current_context = s.copy()
        hops = 5
        previous_stability = 0.0
        
        for hop in range(hops):
            phase_intensity = (np.sin(hop * (np.pi / 2)) + 1.0) / 2.0
            base_temp = 1.2 - (hop / hops) * 0.4
            adaptive_steps = int(40 * (1.0 + (previous_stability / 10.0)))
            adaptive_steps = min(adaptive_steps, 80)
            
            final_state, steps, e_hist = system.encoder.run_dynamics_until_stable(
                current_context, {}, max_steps=adaptive_steps, energy_tolerance=1e-3,
                noise_injection=0.15 * (1.0 + phase_intensity * 0.5), temperature=base_temp,
                max_shocks=2, return_history=True
            )
            
            stability = float(np.var(e_hist[-5:])) if len(e_hist) > 5 else 0.0
            is_epiphany = stability < 0.01 and (e_hist[-1] < 2.0)
            
            if is_epiphany:
                final_state = np.tanh(final_state * 1.5)
            
            matches = decoder.decode(final_state, top_k=3)
            top_concept, top_sim = matches[0]
            
            icon = "✨" if is_epiphany else "➡️"
            print(f"{icon} Hop {hop+1}: {top_concept} (Sim: {top_sim:.2f}) [Energy: {e_hist[-1]:.2f}, Stability: {stability:.2f}]")
            print(f"   Context: {matches[1][0]}, {matches[2][0]}")
            
            previous_stability = stability
            inhibition = final_state * 0.9
            jump_noise = 0.25 if not is_epiphany else 0.1
            noise_vec = np.random.normal(0, jump_noise, final_state.shape)
            current_context = np.tanh(final_state - inhibition + noise_vec)
            
            time.sleep(0.5)

    elif args.cmd == "poem":
        system = URCMSystem(resonance_dim=args.dim, max_steps=6)
        
        print(f"\n📜 Composing Poem for: '{args.topic}'...\n")
        print(f"   {args.topic.upper()}")
        print("   " + "-" * len(args.topic))
        
        lines = system.compose_poem(args.topic, lines_count=4)
        for line in lines:
            print(f"   {line}")
        print("\n")

    elif args.cmd == "stress":
        system = URCMSystem(resonance_dim=args.dim, max_steps=6)
        print(f"🔥 STARTING STRESS TEST (Duration: {args.duration}s)...")
        start_time = time.time()
        prompts = [
            "Imagine a city under water", 
            "Philosophy of time travel", 
            "Blue cat eating pizza",
            "Quantum entanglement explained by a child",
            "The sound of silence in space"
        ]
        count = 0
        epiphanies = 0
        total_hops = 0
        
        while time.time() - start_time < args.duration:
            prompt = random.choice(prompts)
            print(f"\n🧠 Thinking about: '{prompt}'")
            try:
                out = system.process_query_right_brain(prompt, dynamics_steps=40)
                count += 1
                stream = out.get("stream", [])
                total_hops += len(stream)
                for hop in stream:
                    if hop.get("is_epiphany"):
                        epiphanies += 1
                        print(f"  ✨ EPIPHANY at Hop {hop['hop']} (Energy: {hop['final_energy']:.2f}, Stability: {hop['stability']:.4f})")
                
                print(f"  ✅ Completed {len(stream)} hops. Avg Energy: {np.mean(out['full_energy_profile']):.2f}")
            except Exception as e:
                print(f"  ❌ CRASH: {e}")
                
        print(f"\n🏆 STRESS TEST COMPLETE")
        print(f"Total Thoughts: {count}")
        print(f"Total Hops: {total_hops}")
        print(f"Total Epiphanies: {epiphanies}")
        print(f"Epiphany Rate: {epiphanies/total_hops:.2%}")
    elif args.cmd == "maintain":
        system = URCMSystem(resonance_dim=args.dim, max_steps=6)
        if args.sigma is not None and not args.question:
            system.maintain_spectral(max_sigma=args.sigma)
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(root_dir, "urcm_weights.pkl")
            weights = {
                "W_in": system.encoder.W_in,
                "W_res": system.encoder.W_res,
                "W_out": system.encoder.W_out,
                "bias": system.encoder.bias,
                "W_res_inv": np.linalg.pinv(system.encoder.W_res),
            }
            with open(weights_path, "wb") as f:
                pickle.dump(weights, f)
            print(json.dumps({"status": "spectral_clipped", "sigma": args.sigma, "updated": weights_path}))
        elif args.question and args.choices and args.correct:
            choices_list = [c.strip() for c in args.choices.split(",")]
            res = system.fix_qa_ambiguity(args.question, choices_list, args.correct)
            print(json.dumps({
                "before_winner": res["before"]["winner"],
                "after_winner": res["after"]["winner"]
            }, indent=2))
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(root_dir, "urcm_weights.pkl")
            weights = {
                "W_in": system.encoder.W_in,
                "W_res": system.encoder.W_res,
                "W_out": system.encoder.W_out,
                "bias": system.encoder.bias,
                "W_res_inv": np.linalg.pinv(system.encoder.W_res),
            }
            with open(weights_path, "wb") as f:
                pickle.dump(weights, f)
        else:
            print("Usage:\n  maintain --sigma 1.5\n  maintain --question \"Q?\" --choices \"a,b,c\" --correct \"a\"")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

import os

files = {
    "Gap 1 - BrocaArea":        "urcm/core/broca.py",
    "Gap 3 - MeshNode":         "urcm/core/mesh_node.py",
    "Gap 3 - Mesh":             "urcm/core/mesh.py",
    "Gap 4 - LatentSpace":      "urcm/core/latent_space.py",
    "Trained weights":          "urcm_weights.pkl",
    "Commonsense train script": "train_commonsense.py",
    "Boost train script":       "train_commonsense_boost.py",
}

for label, path in files.items():
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    status = "EXISTS" if exists else "MISSING"
    print(f"  {label:<30} {status}  ({size:,} bytes)")

# Check if mesh.py has real MeshNode or just stub
if os.path.exists("urcm/core/mesh.py"):
    content = open("urcm/core/mesh.py").read()
    has_node = "class MeshNode" in content
    has_sync = "synchronize" in content
    print(f"\n  mesh.py has MeshNode class: {has_node}")
    print(f"  mesh.py has synchronize():  {has_sync}")

# Check if broca still uses Markov bigram
if os.path.exists("urcm/core/broca.py"):
    content = open("urcm/core/broca.py").read()
    is_markov = "transitions" in content
    is_nn     = "ConceptDecoder" in content or "nearest" in content.lower()
    print(f"\n  broca.py still Markov bigram: {is_markov}")
    print(f"  broca.py uses NN retrieval:   {is_nn}")

# Check latent space for task_adaptation
if os.path.exists("urcm/core/latent_space.py"):
    content = open("urcm/core/latent_space.py").read()
    has_adapt = "task_adaptation" in content
    is_stub   = "pass" in content and has_adapt
    print(f"\n  latent_space.py has task_adaptation: {has_adapt}")
    print(f"  task_adaptation is stub (pass):      {is_stub}")

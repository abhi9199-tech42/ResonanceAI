
import numpy as np
import time
from typing import List, Dict, Optional, Set
from urcm.core.data_models import MeshSignal

class MeshNode:
    """
    Represents a single node in the decentralized cognitive mesh.
    
    A MeshNode encapsulates a local reasoning instance (abstracted here) and participates
    in the global resonance through privacy-preserving signal exchange.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.neighbors: List['MeshNode'] = []
        
        # Local State
        self.current_mu: float = 0.0
        self.previous_mu: float = 0.0
        self.phase: float = np.random.uniform(0, 2*np.pi)
        
        # Network Dynamics Parameters
        self.coupling_strength: float = 0.1
        self.learning_rate: float = 0.01
        
        # Fault Tolerance & Status
        self.is_active: bool = True
        self.health_score: float = 1.0
        self.error_history: List[str] = []
        
        # Security / Privacy
        self.trusted_neighbors: Set[str] = set()

    def connect(self, node: 'MeshNode'):
        """ Establish a bidirectional connection. """
        if node not in self.neighbors and node != self:
            self.neighbors.append(node)
            self.trusted_neighbors.add(node.node_id)
            # reciprocity handled by caller usually, but let's ensure it for simplicity
            if self not in node.neighbors:
                node.connect(self)

    def disconnect(self, node: 'MeshNode'):
        """ Sever connection. """
        if node in self.neighbors:
            self.neighbors.remove(node)
            if node.node_id in self.trusted_neighbors:
                self.trusted_neighbors.remove(node.node_id)
            if self in node.neighbors:
                node.disconnect(self)

    def update_local_state(self, mu: float, phase: float = None):
        """
        Update the node's state based on local computation.
        Args:
            mu: Current resonance value calculated by local engine.
            phase: Current oscillation phase (optional update).
        """
        self.previous_mu = self.current_mu
        self.current_mu = mu
        if phase is not None:
            self.phase = phase % (2 * np.pi)

    def broadcast_signal(self, signal_type: str = "sync") -> int:
        """
        Broadcasts the current state (delta_mu, phase) to all neighbors.
        Returns number of successful transmissions.
        """
        if not self.is_active:
            return 0
            
        delta_mu = self.current_mu - self.previous_mu
        
        # Construct Privacy-Preserving Signal
        # Only contains scalar metrics, no semantic content.
        signal = MeshSignal(
            sender_id=self.node_id,
            delta_mu=delta_mu,
            phase_alignment=self.phase,
            timestamp=time.time(),
            signal_type=signal_type
        )
        
        sent_count = 0
        for neighbor in self.neighbors:
            try:
                neighbor.receive_signal(signal)
                sent_count += 1
            except Exception as e:
                self._log_error(f"Failed to send to {neighbor.node_id}: {e}")
                
        return sent_count

    def receive_signal(self, signal: MeshSignal):
        """
        Process an incoming signal from a neighbor using synchronization dynamics.
        """
        if not self.is_active:
            return

        # 1. Validation & Fault Tolerance
        if not self._validate_signal(signal):
            return

        # 2. Process Signal (Sync Dynamics)
        # We adjust our phase to align with neighbors who have positive resonance trends (high delta_mu)
        # Kuramoto-like adjustment: dTheta = K * weight * sin(theta_j - theta_i)
        
        # Weight depends on the sender's delta_mu. 
        # If neighbor is gaining insight (positive delta_mu), we listen more.
        # If neighbor is confused (negative delta_mu), we ignore or decouple.
        weight = max(0.0, signal.delta_mu) * self.coupling_strength
        
        phase_diff = signal.phase_alignment - self.phase
        adjustment = weight * np.sin(phase_diff)
        
        self.phase += adjustment
        self.phase %= (2 * np.pi)
        
        # 3. Mu Synchronization (Optional)
        # We might boost our own exploration if neighbors are finding things
        # But we don't directly copy mu, as it must be earned locally.
        
    def _validate_signal(self, signal: MeshSignal) -> bool:
        """
        Validates signal integrity and enforces security/privacy constraints.
        """
        # Security: Is sender known?
        if signal.sender_id not in self.trusted_neighbors:
             # In a real mesh, we might allow new discovery, but for now strict.
             self._log_error(f"Received signal from untrusted source: {signal.sender_id}")
             return False

        # Fault Tolerance: Check timestamp sanity
        current_time = time.time()
        if signal.timestamp > current_time + 5.0: # Future timestamp
            self._log_error(f"Signal from {signal.sender_id} has future timestamp")
            self.health_score -= 0.1
            return False
            
        if signal.timestamp < current_time - 30.0: # Too old
            return False # Just ignore, don't penalize heavily
            
        # Data Integrity
        if not np.isfinite(signal.delta_mu) or not np.isfinite(signal.phase_alignment):
            self._log_error(f"Signal from {signal.sender_id} contains invalid values")
            self.health_score -= 0.2
            return False
            
        return True

    def _log_error(self, message: str):
        self.error_history.append(f"{time.time()}: {message}")
        # Cap log size
        if len(self.error_history) > 100:
            self.error_history.pop(0)

class MeshNetwork:
    """
    Manager to simulate/orchestrate a collection of MeshNodes.
    """
    def __init__(self):
        self.nodes: Dict[str, MeshNode] = {}
        
    def add_node(self, node: MeshNode):
        self.nodes[node.node_id] = node
        
    def create_fully_connected(self):
        """Connect all existing nodes to each other."""
        node_list = list(self.nodes.values())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node_list[i].connect(node_list[j])

    def connect_random_neighbors(self, k: int = 4):
        """
        Connects nodes in a random topology for better scalability than fully connected.
        Each node will attempt to connect to 'k' other random nodes.
        """
        import random
        node_list = list(self.nodes.values())
        n = len(node_list)
        if n < 2:
            return

        for node in node_list:
            # Simple random graph generation
            candidates = random.sample(node_list, min(k, n - 1))
            for candidate in candidates:
                if candidate != node:
                    node.connect(candidate)

    def step_broadcast(self) -> int:
        """
        Trigger all nodes to broadcast.
        Returns total number of successful signal propagations.
        """
        total_signals = 0
        for node in self.nodes.values():
            total_signals += node.broadcast_signal()
        return total_signals

    def synchronize(self) -> float:
        """
        Run one full mesh synchronization round.

        Each active node broadcasts its signal; all neighbours receive it.
        Returns the Kuramoto order parameter r ∈ [0, 1]:
          r = 0  →  fully incoherent (nodes out of phase)
          r = 1  →  full synchronization

        This is the single number that measures global mesh coherence.
        """
        node_list = [n for n in self.nodes.values() if n.is_active]
        if not node_list:
            return 0.0

        # Collect signal dicts from active nodes
        signal_map: dict = {}
        for node in node_list:
            signal_map[node.node_id] = {
                "node_id":   node.node_id,
                "delta_mu":  node.current_mu - node.previous_mu,
                "phase":     node.phase,
                "mu":        node.current_mu,
            }

        # Distribute to neighbours
        for node in node_list:
            incoming = [
                signal_map[nb.node_id]
                for nb in node.neighbors
                if nb.node_id in signal_map and nb.is_active
            ]
            # Kuramoto phase update
            coupling = 0.15
            for sig in incoming:
                phase_diff = sig["phase"] - node.phase
                weight = max(0.0, sig["delta_mu"])
                node.phase += coupling * weight * np.sin(phase_diff)
            node.phase %= (2 * np.pi)

        # Kuramoto order parameter: r = |mean(e^{i*theta})|
        phases = np.array([n.phase for n in node_list])
        r = float(np.abs(np.mean(np.exp(1j * phases))))
        return r

    def get_consensus(self) -> "np.ndarray":
        """
        Compute consensus resonance state as the mean of all active node states.
        Returns a 1-D numpy array of shape (resonance_dim,), or None if no
        active nodes have processed any text yet.

        This does NOT expose individual node vectors — it returns a population
        average, which is analogous to a global semantic centre-of-mass.
        """
        import numpy as np
        active = [n for n in self.nodes.values()
                  if n.is_active and n.current_mu != 0.0]
        if not active:
            return None
        # Stack all current states and mean-pool
        states = np.stack([
            np.full(n.resonance_dim if hasattr(n, "resonance_dim") else 256,
                    n.current_mu)
            for n in active
        ])
        return np.mean(states, axis=0)

    def broadcast(self, text: str) -> dict:
        """
        Send the same text to every active node and return their states.
        Returns {node_id: ResonanceState} — useful for testing and inspection.
        Requires nodes to be MeshNode instances (from mesh_node.py).
        """
        results = {}
        for node in self.nodes.values():
            if not node.is_active:
                continue
            try:
                # Works with both old MeshNode (mesh.py) and new MeshNode (mesh_node.py)
                if hasattr(node, "process"):
                    results[node.node_id] = node.process(text)
                else:
                    node.update_local_state(mu=0.5)
                    results[node.node_id] = node.current_mu
            except Exception as e:
                results[node.node_id] = None
        return results



import pytest
import numpy as np
import time
from dataclasses import dataclass
from hypothesis import given, strategies as st, settings, HealthCheck

from urcm.core.mesh import MeshNode, MeshNetwork
from urcm.core.data_models import MeshSignal

class TestMeshArchitecture:
    """
    Validates decentralized mesh architecture properties.
    Properties 7 (Privacy) and 12 (Fault Tolerance).
    """

    def test_property_privacy_preservation(self):
        """
        Property 7: Decentralized Mesh Privacy Preservation.
        REQ 5.2, 5.3: Ensure encoded semantic vectors are NEVER transmitted.
        Only scalar control signals (mu, phase) are allowed.
        """
        sender = MeshNode("sender")
        receiver = MeshNode("receiver")
        
        sender.connect(receiver)
        
        # Simulate local state with rich semantic data (which implies vector existence in reasoning engine)
        # We manually update the node's simple state
        sender.update_local_state(mu=0.8, phase=1.5)
        
        # Mock the receiver to capture the raw object transmitted
        original_receive = receiver.receive_signal
        captured_signals = []
        
        def spy_receive(signal):
            captured_signals.append(signal)
            original_receive(signal)
            
        receiver.receive_signal = spy_receive
        
        # Broadcast
        sender.broadcast_signal()
        
        assert len(captured_signals) == 1
        signal = captured_signals[0]
        
        # Strict Content Validation
        # 1. Check type
        assert isinstance(signal, MeshSignal)
        
        # 2. Verify NO semantic payload attributes exist checks via reflection
        # MeshSignal is a dataclass, so __dict__ keys are the only fields.
        keys = signal.__dict__.keys()
        forbidden_terms = ['vector', 'phoneme', 'text', 'embedding', 'latent', 'semantic']
        
        for key in keys:
            for term in forbidden_terms:
                assert term not in key.lower(), f"Privacy Leak: Signal contains field '{key}' matching forbidden term '{term}'"
                
        # 3. Verify values are scalars (or basic ID strings)
        assert isinstance(signal.delta_mu, float)
        assert isinstance(signal.phase_alignment, float)
        assert isinstance(signal.timestamp, float)

    def test_property_fault_tolerance_untrusted_source(self):
        """
        Property 12: Mesh Fault Tolerance (Untrusted Source).
        REQ 5.5: Nodes must reject signals from unknown/untrusted peers.
        """
        node = MeshNode("secure_node")
        attacker = MeshNode("attacker")
        
        # Attacker tries to send signal without connection
        # (Simulating by manually invoking receive, as attacker isn't in neighbors list implies no physical link in this model, 
        # but trusted_neighbors check handles logical Trust)
        
        fake_signal = MeshSignal(
            sender_id="attacker",
            delta_mu=0.1,
            phase_alignment=0.0,
            timestamp=time.time(),
            signal_type="sync"
        )
        
        # Node receives it
        node.receive_signal(fake_signal)
        
        # Should log error and NOT update phase
        assert "untrusted" in node.error_history[-1]
        
    def test_property_fault_tolerance_malformed_data(self):
        """
        Property 12: Mesh Fault Tolerance (Bad Data).
        REQ 10.4: System robustness against corrupted inputs.
        """
        node = MeshNode("robust_node")
        friend = MeshNode("friend")
        node.connect(friend)
        
        # 1. Future Timestamp
        future_sig = MeshSignal(
            sender_id="friend",
            delta_mu=0.1,
            phase_alignment=0.0,
            timestamp=time.time() + 9999,
            signal_type="sync"
        )
        node.receive_signal(future_sig)
        assert "future timestamp" in node.error_history[-1]
        assert node.health_score < 1.0 # Penalized
        
        # 2. NaN values
        nan_sig = MeshSignal(
            sender_id="friend",
            delta_mu=np.nan,
            phase_alignment=0.0,
            timestamp=time.time(),
            signal_type="sync"
        )
        node.receive_signal(nan_sig)
        assert "invalid values" in node.error_history[-1]

    @given(st.lists(st.floats(min_value=0, max_value=6.28), min_size=3, max_size=5))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow], max_examples=10)
    def test_synchronization_convergence(self, initial_phases):
        """
        Validates basic synchronization dynamics.
        Though not strictly a 'property' of class invariants, it validates System Behavior.
        """
        network = MeshNetwork()
        nodes = []
        for i, p in enumerate(initial_phases):
            n = MeshNode(f"n{i}")
            n.phase = p
            # Set positive change to encourage syncing
            n.current_mu = 1.0 
            n.previous_mu = 0.9 
            network.add_node(n)
            nodes.append(n)
            
        network.create_fully_connected()
        
        def order_parameter(ns):
            # Kuramoto order parameter r
            # r = |1/N * sum(e^(i*theta))|
            phasors = [np.exp(1j * n.phase) for n in ns]
            z = np.sum(phasors) / len(ns)
            return np.abs(z)
            
        r_initial = order_parameter(nodes)
        
        # Run simulation steps
        for _ in range(10):
            network.step_broadcast()
            
        r_final = order_parameter(nodes)
        
        # If already synced (r close to 1), it should stay. 
        # If not, it should improve or stay similar (depends on coupling).
        # We generally expect improvement for identical frequencies (0 natural freq here).
        
        if r_initial < 0.9:
            assert r_final >= r_initial - 0.05, "Synchronization significantly degraded"
            # In a perfect Kuramoto model it should strictly increase. 
            # With discrete steps and small coupling, allow margin.


    def test_scalability_mechanism(self):
        """
        Validates scalability mechanism (random connections).
        REQ 5.1/scalability: Ensure we don't need fully connected graph.
        """
        network = MeshNetwork()
        node_count = 20
        for i in range(node_count):
            network.add_node(MeshNode(f"n{i}"))
            
        # Connect with low degree k=3
        network.connect_random_neighbors(k=3)
        
        # Verify not fully connected (which would be N*(N-1)/2 = 190 edges)
        # Here we expect roughly N*k edges (directed) or N*k/2 undirected, 
        # but since connect is bidirectional in logic, let's just check neighbors count.
        
        total_neighbors = sum(len(n.neighbors) for n in network.nodes.values())
        
        # Each node attempts to add 3. 
        # Reciprocity might increase it, but it shouldn't be fully connected (19 per node).
        avg_degree = total_neighbors / node_count
        assert avg_degree < node_count - 1 # Should be much less
        assert avg_degree >= 3 # At least what we asked for

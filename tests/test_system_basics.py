import numpy as np
from urcm.core.system import URCMSystem

def test_encoder_outputs_correct_dimension():
    system = URCMSystem(resonance_dim=32, max_steps=6)
    path = system.process_query("test query")
    vec = path.final_state.resonance_vector
    assert vec.shape[0] == 32

def test_metrics_are_captured():
    system = URCMSystem(resonance_dim=32, max_steps=6)
    _ = system.process_query("hello world")
    assert len(system.status.get("metrics_history", [])) >= 1
    last = system.status["metrics_history"][-1]
    assert {"mu","rho","chi"}.issubset(last.keys())

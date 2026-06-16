import pytest
from urcm.core.reasoning import ReasoningEngine
from urcm.core.system import URCMSystem
from urcm.core.symbolic_engine import SymbolicEngine

def test_transitive_and_syllogism_gates():
    engine = ReasoningEngine()
    traj1 = engine.solve("rain", [], [{"type": "TRANSITIVE", "operands": ["rain", "water", "wet"], "weight": 1.0}], steps=3)
    assert isinstance(traj1, list)
    assert len(traj1) >= 1
    traj2 = engine.solve("socrates", [], [{"type": "SYLLOGISM", "operands": ["socrates", "man", "mortal"], "weight": 1.0}], steps=3)
    assert isinstance(traj2, list)
    assert len(traj2) >= 1

def test_tollens_and_ponens_gates():
    engine = ReasoningEngine()
    traj1 = engine.solve("attack", [], [{"type": "PONENS", "operands": ["attack", "defense"], "weight": 1.5}], steps=3)
    assert isinstance(traj1, list)
    assert len(traj1) >= 1
    traj2 = engine.solve("attack", [], [{"type": "TOLLENS", "operands": ["attack", "defense"], "weight": 1.0}], steps=3)
    assert isinstance(traj2, list)
    assert len(traj2) >= 1

def test_concept_creation_one_shot_and_zero_shot():
    engine = ReasoningEngine()
    ok1 = engine.add_concept_from_examples("giraffe", ["animal", "neck"])
    assert isinstance(ok1, bool)
    ok2 = engine.create_zero_shot_concept("zebra", ["horse", "stripes"])
    assert isinstance(ok2, bool)
    assert "zebra" in engine.concept_map

def test_humor_and_beauty_scores():
    engine = ReasoningEngine()
    h = engine.detect_humor(["lawyer", "banana"])
    assert isinstance(h, float)
    assert 0.0 <= h <= 1.0
    b = engine.beauty_score("art")
    assert isinstance(b, float)

def test_counterfactual_and_hypothesis():
    engine = ReasoningEngine()
    cf = engine.run_counterfactual("gravity", "weak", 1.0)
    assert isinstance(cf, list)
    assert len(cf) >= 1
    hp = engine.form_hypothesis("attack", "defense", 1.0)
    assert isinstance(hp, list)
    assert len(hp) >= 1

def test_symbolic_math_and_sequence():
    urcm = URCMSystem()
    ok, res, err = urcm.evaluate_math("15 + 27")
    assert ok
    assert res == 42
    se = SymbolicEngine()
    nxt = se.infer_next_in_sequence([2, 4, 6, 8])
    assert nxt == 10.0
    nxt2 = se.infer_next_in_sequence([3, 9, 27])
    assert nxt2 == 81.0

def test_system_validation_and_execution():
    urcm = URCMSystem()
    before = urcm.status["processed_count"]
    path = urcm.process_query("hello world")
    after = urcm.status["processed_count"]
    assert after == before + 1
    checks = urcm.validate_system()
    assert checks["overall_health"] is True

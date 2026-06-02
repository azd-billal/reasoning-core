from itertools import product
import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(repo_root))

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from reasoning_core.tasks.swig import Swig, SwigCounterfactualEngine

def make_bn(edges, cpds):
    bn = DiscreteBayesianNetwork(edges)
    bn.add_cpds(*cpds)
    assert bn.check_model()
    return bn


def run_case(name, bn, interventions, target, factual_evidence, n_round=4):
    swig = Swig.from_bn(bn, interventions)
    swig.validate_swig_metadata()
    assert swig.check_model()
    engine = SwigCounterfactualEngine(swig, max_response_functions=2_000_000)
    result = engine.query(target, factual_evidence, n_round=n_round)
    assert abs(sum(result.values()) - 1.0) < 1e-8
    print(f"\n=== {name} ===")
    print("Interventions:", interventions)
    print("Factual evidence:", factual_evidence)
    print("Target SWIG node:", swig.random_node_for(target))
    print("Counterfactual distribution:", result)
    print("P(evidence):", round(engine.last_evidence_probability, n_round))
    print("Response-space size:", engine.response_space_size())
    print("Trace:")
    print("\n".join(engine.trace))
    return swig, engine, result

cpd_x = TabularCPD(variable="X", variable_card=2, values=[[0.5], [0.5]], state_names={"X": [0, 1]})
cpd_z = TabularCPD(variable="Z", variable_card=2, values=[[0.5], [0.5]], state_names={"Z": [0, 1]})
cpd_y_and = TabularCPD(
                variable="Y",
                variable_card=2,
                values=[[1.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]],
                evidence=["X","Z"],
                evidence_card=[2,2],
                state_names={"Y": [0,1], "X": [0, 1], "Z": [0, 1]},
                )

bn_and = make_bn(
                [("X","Y"),("Z","Y")],
                [cpd_x,cpd_z,cpd_y_and])


_, _, result_test_Z_1_X_0 = run_case(
    "When Z is known present with X forced to 0",
    bn_and,
    interventions={"X": 0},
    target="Y",
    factual_evidence={"X": 1, "Z": 1, "Y": 1},
)

_ , _, result_test_Z_1_X_1 = run_case(
    "When Z is known present with X forced to 1",
    bn_and,
    interventions={"X": 1},
    target="Y",
    factual_evidence={"X": 1, "Z": 1, "Y": 1},
)

_, _, result_test_Z_0 = run_case(
    "When Z is known absent",
    bn_and,
    interventions={"X": 1},
    target="Y",
    factual_evidence={"X": 1, "Z": 0, "Y": 0},
)

_, _, result_test_Z_unobserved = run_case(
    "When Z is unobserved",
    bn_and,
    interventions={"X": 1},
    target="Y",
    factual_evidence={"X": 0, "Y": 0},
)

def test_Z_0():
    assert(result_test_Z_0          == {0: 1.0, 1:0.0})

def test_Z_1_X_0():
    assert(result_test_Z_1_X_0      == {0: 1.0, 1:0.0})

def test_Z_1_X_0():
    assert(result_test_Z_1_X_1      == {0: 0.0, 1:1.0})

def test_Z_unobserved():
    assert(result_test_Z_unobserved == {0: 0.5, 1:0.5})

cpd_z = TabularCPD("Z",2,[[0.6],[0.4]],state_names={"Z": [0,1]})

cpd_x_given_Z = TabularCPD(
        "X",
        2,
        [[0.1,0.9],[0.9,0.1]],
        evidence = ["Z"], evidence_card=[2],
        state_names={"X":[0,1], "Z": [0,1]},)

p_y1 = [0.85,0.20,0.80,0.15]
cpd_y_given_xz = TabularCPD(
    "Y",
    2,
    [[1-p for p in p_y1], p_y1],
    evidence=["X", "Z"], evidence_card=[2,2],
    state_names={"Y": [0,1], "X": [0,1], "Z": [0,1]},
)

bn_simpson = make_bn(
    [('Z','X'),('Z','Y'),('X','Y')],
    [cpd_z,cpd_x_given_Z,cpd_y_given_xz],
)

_,_,result_simpson_unobs_z = run_case("Simpson paradox with unobserved Z",
    bn_simpson, 
    interventions={"X":0},
    target="Y",
    factual_evidence={"X":1,"Y":1})

_,_,result_simpson_unobs_z_reversed_xy = run_case("Simpson paradox with unobserved Z but X and Y reversed",
    bn_simpson, 
    interventions={"X":1},
    target="Y",
    factual_evidence={"X":0,"Y":0})

_,_, result_simpson_z0 = run_case("Simpson within Z=0",
    bn_simpson, 
    interventions={"X":0},
    target="Y",
    factual_evidence={"X":1,"Y":1,"Z":0})

def test_simpson():
    assert(0.825 < result_simpson_unobs_z[1] < 0.875) # arbitrary step : +- 0.025 
    assert(0.125 < result_simpson_unobs_z_reversed_xy[1] < 0.175) # arbitrary step : +- 0.025
    assert(result_simpson_z0[1] == 0.85)
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


cpd_v = TabularCPD(
    "V",
    2,
    [[0.5],[0.5]],
    state_names={'V': [0,1]},
)

cpd_x = TabularCPD(
    "X",
    2,
    [[0.2,0.1],[0.8,0.9]],
    evidence=["V"],evidence_card=[2],
    state_names={'X': [0,1], 'V': [0,1]},
)

cpd_w = TabularCPD(
    "W",
    2,
    [[0.5,0.8],[0.5,0.2]],
    evidence=["X"],evidence_card=[2],
    state_names={'W': [0,1],'X': [0,1]},
)

cpd_y = TabularCPD(
    "Y",
    2,
    [[0.3,0.4,0.5,0.6],[0.7,0.6,0.5,0.4]],
    evidence=["V","W"],evidence_card=[2, 2],
    state_names={'Y': [0,1],'V': [0,1],'W': [0,1]},
)

cpd_z = TabularCPD(
    "Z",
    2,
    [[0.9,0.7,0.5,0.3],[0.1,0.3,0.5,0.7]],
    evidence=["X","Y"],evidence_card=[2, 2],
    state_names={'Z': [0,1],'X': [0,1],'Y': [0,1]},
)

bn_scenario = make_bn(
    [("V", "X"), ("V", "Y"), ("X", "W"), ("X", "Z"), ("W", "Y"), ("Y", "Z")],
    [cpd_v, cpd_w, cpd_x, cpd_y, cpd_z],
)

@pytest.mark.parametrize(
    "name, bn, interventions, evidence, expected",
    [
        (
            "A1: do(X=1)",
            bn_scenario,
            {"X": 1},
            {"V": 0, "W": 0, "X": 0, "Y": 0, "Z": 0},
            {0: 0.364, 1: 0.636},
        ),
        (
            "A2: do(W=1)",
            bn_scenario,
            {"W": 1},
            {"V": 0, "W": 0, "X": 1, "Y": 1, "Z": 1},
            {0: 0.38, 1: 0.62},
        ),
        (
            "A3: do(W=1)",
            bn_scenario,
            {"W": 1},
            {"V": 1, "W": 0, "X": 0, "Y": 1, "Z": 0},
            {0: 0.82, 1: 0.18},
        ),
    ],
)

def test_counterfactuals(name, bn, interventions, evidence, expected):
    _, _, result = run_case(
        name,
        bn,
        interventions=interventions,
        target="Z",
        factual_evidence=evidence,
    )
    assert result == pytest.approx(expected, rel=1e-6)

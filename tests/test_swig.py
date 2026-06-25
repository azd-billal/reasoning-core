from itertools import product
import sys
from pathlib import Path
import pytest

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
    assert bn.check_model()
    swig = Swig(bn=bn)
    swig.generate_225(intervention=interventions, target=target, observations=factual_evidence)
    engine = swig.engine
    result, _ = engine.compute_cbn_225()
    assert abs(sum(result.values()) - 1.0) < 1e-8
    print(f"\n=== {name} ===")
    print("Interventions:", interventions)
    print("Factual evidence:", factual_evidence)
    print("Target SWIG node:", target)
    print("Counterfactual distribution:", result)
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
    evidence=["X", "Z"],
    evidence_card=[2, 2],
    state_names={"Y": [0, 1], "X": [0, 1], "Z": [0, 1]},
)

bn_and = make_bn(
    [("X", "Y"), ("Z", "Y")],
    [cpd_x, cpd_z, cpd_y_and],
)


cpd_z_25_75 = TabularCPD(variable="Z", variable_card=2, values=[[0.25], [0.75]], state_names={"Z": [0, 1]})
bn_and_2 = make_bn(
    [("X", "Y"), ("Z", "Y")],
    [cpd_x, cpd_z_25_75, cpd_y_and],
)


cpd_z = TabularCPD("Z", 2, [[0.6], [0.4]], state_names={"Z": [0, 1]})

cpd_x_given_Z = TabularCPD(
    "X", 2,
    [[0.1, 0.9], [0.9, 0.1]],
    evidence=["Z"], evidence_card=[2],
    state_names={"X": [0, 1], "Z": [0, 1]},
)

cpd_y_given_xz = TabularCPD(
    "Y", 2,
    [[0.15, 0.80, 0.20, 0.85],
     [0.85, 0.20, 0.80, 0.15]],
    evidence=["X", "Z"], evidence_card=[2, 2],
    state_names={"Y": [0, 1], "X": [0, 1], "Z": [0, 1]},
)

bn_simpson = make_bn(
    [("Z", "X"), ("Z", "Y"), ("X", "Y")],
    [cpd_z, cpd_x_given_Z, cpd_y_given_xz],
)


@pytest.mark.parametrize(
    "name, bn, interventions, target, evidence, expected",
    [
        (
            "And Z=0",
            bn_and,
            {"X": 1},
            "Y",
            {"X": 1, "Z": 0, "Y": 0},
            {0: 1.0, 1: 0.0},
        ),
        (
            "And Z=1 X=0",
            bn_and,
            {"X": 0},
            "Y",
            {"X": 1, "Z": 1, "Y": 1},
            {0: 1.0, 1: 0.0},
        ),
        (
            "And Z=1 X=1",
            bn_and,
            {"X": 1},
            "Y",
            {"X": 1, "Z": 1, "Y": 1},
            {0: 0.0, 1: 1.0},
        ),
        (
            "And Z unobserved",
            bn_and,
            {"X": 1},
            "Y",
            {"X": 0, "Y": 0},
            {0: 0.5, 1: 0.5},
        ),
        (
            "And 2 Z unobserved",
            bn_and_2,
            {"X": 1},
            "Y",
            {"X": 0, "Y": 0},
            {0: 0.25, 1: 0.75},
        ),
    ],
)
def test_counterfactuals_and(name, bn, interventions, target, evidence, expected):
    _, _, result = run_case(
        name, bn,
        interventions=interventions,
        target=target,
        factual_evidence=evidence,
    )
    assert result == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "name, bn, interventions, target, evidence, expected, tol",
    [
        (
            "Simpson unobs Z",
            bn_simpson,
            {"X": 0},
            "Y",
            {"X": 1, "Y": 1},
            {0: 0.15, 1: 0.85},
            0.025,
        ),
        (
            "Simpson unobs Z reversed XY",
            bn_simpson,
            {"X": 1},
            "Y",
            {"X": 0, "Y": 0},
            {0: 0.85, 1: 0.15},
            0.025,
        ),
        (
            "Simpson Z=0",
            bn_simpson,
            {"X": 0},
            "Y",
            {"X": 1, "Y": 1, "Z": 0},
            {0: 0.15, 1: 0.85},
            None,
        ),
    ],
)
def test_counterfactuals_simpson(name, bn, interventions, target, evidence, expected, tol):
    _, _, result = run_case(
        name, bn,
        interventions=interventions,
        target=target,
        factual_evidence=evidence,
    )
    if tol is None:
        assert result == pytest.approx(expected, rel=1e-6)
    else:
        assert result == pytest.approx(expected, abs=tol)

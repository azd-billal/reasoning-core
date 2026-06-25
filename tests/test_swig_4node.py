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
    swig.generate_225(intervention=interventions,target=target,observations=factual_evidence)
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


# =============================================================================
# Scenario A: Mediation with confounder (4 nodes)
#   Z -> X -> M -> Y
#   Z -> Y
#
# Y is a deterministic AND of M and Z.
# =============================================================================
cpd_z_a = TabularCPD("Z", 2, [[0.5], [0.5]], state_names={"Z": [0, 1]})

cpd_x_a = TabularCPD(
    "X", 2,
    [[0.9, 0.1], [0.1, 0.9]],
    evidence=["Z"], evidence_card=[2],
    state_names={"X": [0, 1], "Z": [0, 1]},
)

cpd_m_a = TabularCPD(
    "M", 2,
    [[0.9, 0.1], [0.1, 0.9]],
    evidence=["X"], evidence_card=[2],
    state_names={"M": [0, 1], "X": [0, 1]},
)

cpd_y_a = TabularCPD(
    "Y", 2,
    [[1.0, 1.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    evidence=["M", "Z"], evidence_card=[2, 2],
    state_names={"Y": [0, 1], "M": [0, 1], "Z": [0, 1]},
)

bn_scenario_a = make_bn(
    [("Z", "X"), ("X", "M"), ("M", "Y"), ("Z", "Y")],
    [cpd_z_a, cpd_x_a, cpd_m_a, cpd_y_a],
)

# =============================================================================
# Scenario B: Diamond with AND gate (4 nodes)
#   X -> Y -> W
#   X -> Z -> W
#
# W is a deterministic AND of Y and Z.
# =============================================================================
cpd_x_b = TabularCPD("X", 2, [[0.5], [0.5]], state_names={"X": [0, 1]})

cpd_y_b = TabularCPD(
    "Y", 2,
    [[0.8, 0.2], [0.2, 0.8]],
    evidence=["X"], evidence_card=[2],
    state_names={"Y": [0, 1], "X": [0, 1]},
)

cpd_z_b = TabularCPD(
    "Z", 2,
    [[0.7, 0.3], [0.3, 0.7]],
    evidence=["X"], evidence_card=[2],
    state_names={"Z": [0, 1], "X": [0, 1]},
)

cpd_w_b = TabularCPD(
    "W", 2,
    [[1.0, 1.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    evidence=["Y", "Z"], evidence_card=[2, 2],
    state_names={"W": [0, 1], "Y": [0, 1], "Z": [0, 1]},
)

bn_scenario_b = make_bn(
    [("X", "Y"), ("X", "Z"), ("Y", "W"), ("Z", "W")],
    [cpd_x_b, cpd_y_b, cpd_z_b, cpd_w_b],
)

# =============================================================================
# Scenario C: Linear chain (4 nodes)
#   W -> X -> Y -> Z
# =============================================================================
cpd_w_c = TabularCPD("W", 2, [[0.5], [0.5]], state_names={"W": [0, 1]})

cpd_x_c = TabularCPD(
    "X", 2,
    [[0.9, 0.1], [0.1, 0.9]],
    evidence=["W"], evidence_card=[2],
    state_names={"X": [0, 1], "W": [0, 1]},
)

cpd_y_c = TabularCPD(
    "Y", 2,
    [[0.4, 0.2], [0.6, 0.8]],
    evidence=["X"], evidence_card=[2],
    state_names={"Y": [0, 1], "X": [0, 1]},
)

cpd_z_c = TabularCPD(
    "Z", 2,
    [[0.95, 0.15], [0.05, 0.85]],
    evidence=["Y"], evidence_card=[2],
    state_names={"Z": [0, 1], "Y": [0, 1]},
)

bn_scenario_c = make_bn(
    [("W", "X"), ("X", "Y"), ("Y", "Z")],
    [cpd_w_c, cpd_x_c, cpd_y_c, cpd_z_c],
)

@pytest.mark.parametrize(
    "name, bn, interventions, target, evidence, expected",
    [
        # --- B: Diamond ---
        (
            "B1: Diamond Y=0,Z=0,W=0",
            bn_scenario_b,
            {"X": 1},
            "W",
            {"X": 0, "Y": 0, "Z": 0, "W": 0},
            {0: 0.44, 1: 0.56},
        ),
        (
            "B2: Diamond Y=0,Z=1,W=0",
            bn_scenario_b,
            {"X": 1},
            "W",
            {"X": 0, "Y": 0, "Z": 1, "W": 0},
            {0: 0.44, 1: 0.56},
        ),

        # --- A: Mediation avec confounder ---
        (
            "A1: Z unobserved",
            bn_scenario_a,
            {"X": 1},
            "Y",
            {"X": 0, "M": 0, "Y": 0},
            {0: 0.91, 1: 0.09},
        ),
        (
            "A2: Z=1 observed",
            bn_scenario_a,
            {"X": 1},
            "Y",
            {"X": 0, "M": 0, "Y": 0, "Z": 1},
            {0: 0.1, 1: 0.9},
        ),
        (
            "A3: Z=0 observed",
            bn_scenario_a,
            {"X": 1},
            "Y",
            {"X": 0, "M": 0, "Y": 0, "Z": 0},
            {0: 1.0, 1: 0.0},
        ),

        # --- C: Chain ---
        (
            "C1: do(X=1)",
            bn_scenario_c,
            {"X": 1},
            "Z",
            {"X": 0, "Y": 0, "Z": 0},
            {0: 0.31, 1: 0.69},
        ),
        (
            "C2: do(W=1)",
            bn_scenario_c,
            {"W": 1},
            "Z",
            {"W": 0, "X": 0, "Y": 0, "Z": 0},
            {0: 0.326, 1: 0.674},
        ),
    ],
)

def test_counterfactuals(name, bn, interventions, target, evidence, expected):
    _, _, result = run_case(
        name,
        bn,
        interventions=interventions,
        target=target,
        factual_evidence=evidence,
    )

    assert result == pytest.approx(expected, rel=1e-6)
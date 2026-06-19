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

# Case A1: Z unobserved
_, _, result_a1 = run_case(
    "A1: Mediation with confounder, Z unobserved",
    bn_scenario_a,
    interventions={"X": 1},
    target="Y",
    factual_evidence={"X": 0, "M": 0, "Y": 0},
)

# Case A2: Z=1 observed
_, _, result_a2 = run_case(
    "A2: Mediation with confounder, Z=1",
    bn_scenario_a,
    interventions={"X": 1},
    target="Y",
    factual_evidence={"X": 0, "M": 0, "Y": 0, "Z": 1},
)

# Case A3: Z=0 observed
_, _, result_a3 = run_case(
    "A3: Mediation with confounder, Z=0",
    bn_scenario_a,
    interventions={"X": 1},
    target="Y",
    factual_evidence={"X": 0, "M": 0, "Y": 0, "Z": 0},
)


def test_a1_z_unobserved():
    """
    si do(X=1) sachant que X=0 dans le monde réel -> Z= {0=0.9,1=0.1}
    P(M=1|X=1) dans le monde contrefactuel = 0.9
    P(Y=1) = P(Z=1)*P(M=1) = 0.9*0.1 = 0.09
    """
    assert result_a1 == {0: 0.91, 1: 0.09}


def test_a2_z1_observed():
    """
    do(X=1) sachant que X=0 dans le mondé réel
    on observe Z=1
    P(M=1|X=1)= 0.9
    P(Y=1) = P(Z=1)*P(M1)= 1*0.9
    """
    assert result_a2 == {0: 0.1, 1: 0.9}


def test_a3_z0_observed():
    """
    Z=0 et comme Y = Z and M, Y = 0
    """
    assert result_a3 == {0: 1.0, 1: 0.0}


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

_, _, result_b1 = run_case(
    "B1: Diamond with Y=0,Z=0,W=0",
    bn_scenario_b,
    interventions={"X": 1},
    target="W",
    factual_evidence={"X": 0, "Y": 0, "Z": 0, "W": 0},
)

_, _, result_b2 = run_case(
    "B2: Diamond with Y=0,Z=1,W=0",
    bn_scenario_b,
    interventions={"X": 1},
    target="W",
    factual_evidence={"X": 0, "Y": 0, "Z": 1, "W": 0},
)


def test_b1_diamond_y0_z0():
    assert result_b1 == {0: 0.44, 1: 0.56}


def test_b2_diamond_y0_z1():
    assert result_b2 == {0: 0.44, 1: 0.56}


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

_, _, result_c1 = run_case(
    "C1: Chain, do(X=1), X evidence only",
    bn_scenario_c,
    interventions={"X": 1},
    target="Z",
    factual_evidence={"X": 0, "Y": 0, "Z": 0},
)

_, _, result_c2 = run_case(
    "C2: Chain, do(W=1), full evidence",
    bn_scenario_c,
    interventions={"W": 1},
    target="Z",
    factual_evidence={"W": 0, "X": 0, "Y": 0, "Z": 0},
)


def test_c1_chain_do_x():
    """
    do(X=1)
    P(Y) = {0:0.2,1:0.8}
    P(Z=0) = P(Z=0|Y=0)+P(Z=0|Y=1) = 0.2 * 0.95 + 0.8 * 0.15 = 0.31
    P(Z=1) = P(Z=1|Y=0)+P(Z=1|Y=1) = 0.2 * 0.05 + 0.8 * 0.85 = 0.69
    """
    assert result_c1 == {0: 0.31, 1: 0.69}


def test_c2_chain_do_w():
    """
    P(Z=0)  = P(Z=0|Y=0,X=0) + P(Z=0|Y=0,X=1) + P(Z=0|Y=1,X=0) + P(Z=0|Y=1,X=1)
            = 0.1*0.4*0.95   + 0.9*0.2*0.95   + 0.1*0.6*0.15   + 0.9*0.8*0.15 = 0.326 
    P(Z=1)  = P(Z=1|Y=0,X=0) + P(Z=1|Y=0,X=1) + P(Z=1|Y=1,X=0) + P(Z=1|Y=1,X=1)
            = 0.1*0.4*0.05   + 0.9*0.2*0.05   + 0.1*0.6*0.85   + 0.9*0.8*0.85 = 0.674
    """
    assert result_c2 == {0: 0.326, 1: 0.674}

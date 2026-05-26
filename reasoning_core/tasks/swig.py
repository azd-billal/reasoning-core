from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import networkx as nx
import numpy as np

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from reasoning_core.tasks._causal_utils import get_random_DAG
from reasoning_core.template import Config

@dataclass
class SwigConfig(Config):
    num_nodes: int = 5
    graph_density: float = 0.4
    num_latents: int = 1
    cardinality: int = 2
    cpd_low: float = 0.1
    cpd_high: float = 0.9
    random_seed: Optional[int] = None
    max_response_functions: int = 200_000

    def update(self, c: float) -> None:
        self.num_nodes = max(2, int(self.num_nodes * (1 + c)))
        self.graph_density = min(0.8, self.graph_density + 0.1 * c)


@dataclass(frozen=True)
class SwigNodeInfo:
    label: str
    source: str
    kind: str
    value: Optional[Any] = None
    fixed_ancestors: Tuple[str, ...] = ()

    @property
    def is_fixed(self) -> bool:
        return self.kind == "fixed"

    @property
    def is_random(self) -> bool:
        return self.kind == "random"


@dataclass
class SwigSpec:
    source_nodes: Tuple[str, ...] = ()
    source_edges: Tuple[Tuple[str, str], ...] = ()
    interventions: Dict[str, Any] = field(default_factory=dict)
    random_of: Dict[str, str] = field(default_factory=dict)
    fixed_of: Dict[str, str] = field(default_factory=dict)
    source_of: Dict[str, str] = field(default_factory=dict)
    node_info: Dict[str, SwigNodeInfo] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["node_info"] = {label: asdict(info) for label, info in self.node_info.items()}
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SwigSpec":
        raw_info = data.get("node_info", {})
        node_info = {
            str(label): SwigNodeInfo(
                label=str(info["label"]),
                source=str(info["source"]),
                kind=str(info["kind"]),
                value=info.get("value"),
                fixed_ancestors=tuple(info.get("fixed_ancestors", ())),
            )
            for label, info in raw_info.items()
        }
        return cls(
            source_nodes=tuple(map(str, data.get("source_nodes", ()))),
            source_edges=tuple((str(u), str(v)) for u, v in data.get("source_edges", ())),
            interventions={str(k): v for k, v in data.get("interventions", {}).items()},
            random_of={str(k): str(v) for k, v in data.get("random_of", {}).items()},
            fixed_of={str(k): str(v) for k, v in data.get("fixed_of", {}).items()},
            source_of={str(k): str(v) for k, v in data.get("source_of", {}).items()},
            node_info=node_info,
        )


@dataclass(frozen=True)
class _ResponseMechanism:
    source: str
    source_key: Any
    cpd: TabularCPD
    parents: Tuple[str, ...]
    parent_keys: Tuple[Any, ...]
    states: Tuple[Any, ...]
    parent_states: Tuple[Tuple[Any, ...], ...]
    parent_configs: Tuple[Tuple[Any, ...], ...]
    probabilities: Dict[Tuple[Any, ...], Tuple[float, ...]]


def _is_fixed_node(label: str) -> bool:
    return label.startswith("do(")


def _is_counterfactual_node(label: str) -> bool:
    return "(" in label and not label.startswith("do(")


def _get_base_variable(label: str) -> str:
    if _is_fixed_node(label):
        inner = label[3:-1]
        return inner.split("=", 1)[0]
    if "(" in label:
        return label.split("(", 1)[0]
    return label


def _node_to_str(node: Any) -> str:
    return str(node)


def _partition_nodes(graph: nx.DiGraph) -> Tuple[Set[str], Set[str]]:
    all_nodes = set(graph.nodes())
    fixed = {n for n in all_nodes if _is_fixed_node(str(n))}
    return all_nodes - fixed, fixed


if hasattr(nx, "is_d_separator"):
    _nx_d_sep = nx.is_d_separator
elif hasattr(nx, "d_separated"):
    _nx_d_sep = nx.d_separated
else:
    raise ImportError("NetworkX incompatible : aucune fonction de d-separation trouvee.")


def _is_d_separator(G: nx.DiGraph, X: Set, Y: Set, Z: Set) -> bool:
    return _nx_d_sep(G, X, Y, Z)


def _extract_graph(dag_wrapper) -> nx.DiGraph:
    if isinstance(dag_wrapper, nx.DiGraph):
        return dag_wrapper
    for attr in ("nx_dag", "graph"):
        candidate = getattr(dag_wrapper, attr, None)
        if isinstance(candidate, nx.DiGraph):
            return candidate
    if isinstance(dag_wrapper, dict):
        for key in ("nx_dag", "graph"):
            if isinstance(dag_wrapper.get(key), nx.DiGraph):
                return dag_wrapper[key]
    return nx.DiGraph(dag_wrapper)


class Swig(DiscreteBayesianNetwork):
    """
    Reseau Bayesian Causal etendu supportant le formalisme SWIG (Robins & Richardson).

    Deux modes d'utilisation :
    1. Mode source : `generate_random_dag()` cree un DAG causal factuel.
    2. Mode SWIG  : `transform_to_swig()` transforme ce DAG en SWIG avec
       labels lisibles (do(X=v), Y(X=v)) et metadonnees SwigSpec.
    """

    def __init__(
        self,
        config: Optional[SwigConfig] = None,
        *,
        ebunch: Optional[Iterable[Tuple[str, str]]] = None,
        source_bn: Optional["Swig"] = None,
        spec: Optional[SwigSpec] = None,
    ):
        super().__init__(ebunch)
        self.config = config if config is not None else SwigConfig()
        self._rng = random.Random(self.config.random_seed)
        self._np_rng = np.random.default_rng(self.config.random_seed)

        self.source_bn = source_bn.copy() if source_bn is not None else None
        self.spec = spec if spec is not None else SwigSpec()
        self.cardinalities: Dict[str, int] = {}
        self.latents: Set[str] = set()
        self.trace: List[str] = []


    def generate_random_dag(
        self,
        node_names: Optional[List[str]] = None,
        method: str = "erdos",
        required_edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> "Swig":
        n_nodes = len(node_names) if node_names is not None else self.config.num_nodes
        dag_wrapper = get_random_DAG(
            n_nodes=n_nodes,
            edge_prob=self.config.graph_density,
            node_names=node_names,
            latents=(self.config.num_latents > 0),
            seed=self.config.random_seed,
            method=method,
        )
        graph = _extract_graph(dag_wrapper)
        self.add_nodes_from(graph.nodes())
        self.add_edges_from(graph.edges())
        if required_edges:
            self.add_edges_from((str(u), str(v)) for u, v in required_edges)
        self.latents = {str(n) for n in getattr(dag_wrapper, "latents", set())}

        if not nx.is_directed_acyclic_graph(self):
            raise RuntimeError("Le DAG genere contient un cycle.")
        self._generate_cpds()
        if not self.check_model():
            raise RuntimeError("Reseau Bayesien factuel structurellement mal forme.")
        return self

    def _generate_cpds(self) -> None:
        self.cardinalities = {}
        k = self.config.cardinality
        if k * self.config.cpd_low >= 1.0:
            raise ValueError(f"Loi de positivite impossible : k*cpd_low = {k * self.config.cpd_low} >= 1.")
        base_dag = nx.DiGraph()
        base_dag.add_nodes_from(self.nodes())
        base_dag.add_edges_from(self.edges())
        cpds: List[TabularCPD] = []
        for node in nx.topological_sort(base_dag):
            self.cardinalities[node] = k
            parents = list(base_dag.predecessors(node))
            num_parents = len(parents)
            num_combos = max(1, k ** num_parents)
            if k == 2:
                if num_parents == 0:
                    p = self._rng.uniform(self.config.cpd_low, self.config.cpd_high)
                    values = [[1 - p], [p]]
                else:
                    v0, v1 = [], []
                    for _ in range(num_combos):
                        p = self._rng.uniform(self.config.cpd_low, self.config.cpd_high)
                        v0.append(1 - p)
                        v1.append(p)
                    values = [v0, v1]
            else:
                values = [[] for _ in range(k)]
                floor = self.config.cpd_low
                scale = 1.0 - k * floor
                for _ in range(num_combos):
                    raw = self._np_rng.dirichlet([1.0] * k)
                    adjusted = raw * scale + floor
                    for i in range(k):
                        values[i].append(float(adjusted[i]))
            evidence = parents if num_parents > 0 else None
            evidence_card = [k] * num_parents if num_parents > 0 else None
            cpds.append(TabularCPD(
                variable=node, variable_card=k, values=values,
                evidence=evidence, evidence_card=evidence_card,
            ))
        self.add_cpds(*cpds)


    @classmethod
    def from_bn(
        cls,
        bn: "Swig",
        interventions: Mapping[Any, Any],
        *,
        copy_cpds: bool = True,
    ) -> "Swig":
        source_nodes = tuple(str(n) for n in bn.nodes())
        source_edges = tuple((str(u), str(v)) for u, v in bn.edges())
        normalized_interventions = {str(var): value for var, value in interventions.items()}
        missing = set(normalized_interventions) - set(source_nodes)
        if missing:
            raise ValueError(f"Unknown intervention variables: {sorted(missing)}")

        fixed_of = {
            var: cls._fixed_label(var, value) for var, value in normalized_interventions.items()
        }
        temp_graph = nx.DiGraph()
        temp_graph.add_nodes_from(source_nodes)
        temp_graph.add_nodes_from(fixed_of.values())
        for parent, child in source_edges:
            swig_parent = fixed_of[parent] if parent in fixed_of else parent
            temp_graph.add_edge(swig_parent, child)

        fixed_labels = set(fixed_of.values())
        fixed_var_by_label = {label: var for var, label in fixed_of.items()}
        random_of: Dict[str, str] = {}
        node_info: Dict[str, SwigNodeInfo] = {}
        source_of: Dict[str, str] = {}

        for source_var in source_nodes:
            fixed_ancestor_labels = sorted(
                nx.ancestors(temp_graph, source_var).intersection(fixed_labels)
            )
            fixed_ancestor_vars = tuple(
                fixed_var_by_label[label] for label in fixed_ancestor_labels
            )
            label = cls._random_label(source_var, fixed_ancestor_vars, normalized_interventions)
            random_of[source_var] = label
            source_of[label] = source_var
            node_info[label] = SwigNodeInfo(
                label=label, source=source_var, kind="random",
                fixed_ancestors=fixed_ancestor_vars,
            )

        for source_var, label in fixed_of.items():
            source_of[label] = source_var
            node_info[label] = SwigNodeInfo(
                label=label, source=source_var, kind="fixed",
                value=normalized_interventions[source_var],
            )

        if len(node_info) != len(source_nodes) + len(fixed_of):
            raise ValueError("SWIG labels are not unique.")

        spec = SwigSpec(
            source_nodes=source_nodes, source_edges=source_edges,
            interventions=normalized_interventions, random_of=random_of,
            fixed_of=fixed_of, source_of=source_of, node_info=node_info,
        )

        swig = cls(source_bn=bn, spec=spec)
        swig.add_nodes_from(node_info)
        for parent, child in source_edges:
            swig_parent = fixed_of[parent] if parent in fixed_of else random_of[parent]
            swig_child = random_of[child]
            swig.add_edge(swig_parent, swig_child)

        if copy_cpds:
            swig._build_swig_cpds()

        swig.trace.append(
            f"Built SWIG from source BN with interventions {normalized_interventions}."
        )
        return swig

    @classmethod
    def from_random(
        cls,
        *,
        n_nodes: int = 4,
        edge_prob: float = 0.5,
        n_interventions: int = 1,
        cardinality: int = 2,
        seed: Optional[int] = None,
    ) -> "Swig":
        config = SwigConfig(
            num_nodes=n_nodes,
            graph_density=edge_prob,
            num_latents=0,
            cardinality=cardinality,
            random_seed=seed,
        )
        source = cls(config)
        source.generate_random_dag()
        rng = random.Random(seed)
        variables = [str(v) for v in source.nodes()]
        chosen = rng.sample(variables, min(n_interventions, len(variables)))
        interventions = {
            var: rng.randrange(source.cardinalities.get(var, cardinality))
            for var in chosen
        }
        return source.transform_to_swig(interventions)

    @staticmethod
    def _fixed_label(variable: str, value: Any) -> str:
        return f"do({variable}={value})"

    @staticmethod
    def _random_label(
        variable: str, fixed_ancestors: Tuple[str, ...], interventions: Mapping[str, Any]
    ) -> str:
        if not fixed_ancestors:
            return variable
        suffix = ",".join(
            f"{ancestor}={interventions[ancestor]}" for ancestor in sorted(fixed_ancestors)
        )
        return f"{variable}({suffix})"

    def _build_swig_cpds(self) -> None:
        if self.source_bn is None:
            raise ValueError("Cannot build SWIG CPDs without a source BN.")
        self.cardinalities = {}
        for source_var in self.spec.source_nodes:
            if source_var in self.source_bn.cardinalities:
                self.cardinalities[source_var] = self.source_bn.cardinalities[source_var]

        cpds: List[TabularCPD] = []
        for source_var, fixed_label in self.spec.fixed_of.items():
            states = self._states_for(self.source_bn, source_var)
            cpd = self._point_mass_cpd(fixed_label, states, self.spec.interventions[source_var])
            self.cardinalities[fixed_label] = self.cardinalities.get(source_var, len(states))
            cpds.append(cpd)

        for source_var in self.spec.source_nodes:
            source_key = self._source_key_in_source(source_var)
            source_cpd = self.source_bn.get_cpds(source_key)
            if source_cpd is None:
                continue
            if not isinstance(source_cpd, TabularCPD):
                raise TypeError(
                    f"Swig currently copies TabularCPD mechanisms only; "
                    f"got {type(source_cpd)!r} for {source_var}."
                )
            cpds.append(self._redirect_cpd(source_cpd))

        self.add_cpds(*cpds)
        if not self.check_model():
            raise RuntimeError("SWIG Bayesian network is structurally malformed.")

    def _redirect_cpd(self, cpd: TabularCPD) -> TabularCPD:
        source_var = str(cpd.variable)
        swig_var = self.spec.random_of[source_var]
        raw_evidence = list(cpd.variables[1:])
        source_evidence = [str(v) for v in raw_evidence]
        swig_evidence = [
            self.spec.fixed_of[p] if p in self.spec.fixed_of else self.spec.random_of[p]
            for p in source_evidence
        ]
        evidence_card = [int(card) for card in cpd.cardinality[1:]]
        state_names = {swig_var: self._cpd_states(cpd, cpd.variable, int(cpd.variable_card))}
        for raw_parent, swig_parent, parent_card in zip(raw_evidence, swig_evidence, evidence_card):
            state_names[swig_parent] = self._cpd_states(cpd, raw_parent, parent_card)
        kwargs: Dict[str, Any] = {
            "variable": swig_var,
            "variable_card": int(cpd.variable_card),
            "values": np.array(cpd.get_values(), dtype=float, copy=True),
            "state_names": state_names,
        }
        if swig_evidence:
            kwargs["evidence"] = swig_evidence
            kwargs["evidence_card"] = evidence_card
        return TabularCPD(**kwargs)

    @staticmethod
    def _point_mass_cpd(label: str, states: List[Any], value: Any) -> TabularCPD:
        if value not in states:
            raise ValueError(f"Intervention value {value!r} is not in states {states!r}")
        values = np.zeros((len(states), 1))
        values[states.index(value), 0] = 1.0
        return TabularCPD(
            variable=label, variable_card=len(states), values=values,
            state_names={label: states},
        )

    @staticmethod
    def _states_for(bn: DiscreteBayesianNetwork, variable: str) -> List[Any]:
        bn_key = Swig._bn_node_key(bn, variable)
        if hasattr(bn, "states"):
            if bn_key in bn.states:
                return list(bn.states[bn_key])
            if str(variable) in bn.states:
                return list(bn.states[str(variable)])
        cpd = bn.get_cpds(bn_key)
        if cpd is not None and hasattr(cpd, "state_names"):
            if bn_key in cpd.state_names:
                return list(cpd.state_names[bn_key])
            if str(variable) in cpd.state_names:
                return list(cpd.state_names[str(variable)])
        try:
            return list(range(int(bn.get_cardinality(bn_key))))
        except Exception as exc:
            raise ValueError(f"Cannot infer states for variable {variable!r}") from exc

    @staticmethod
    def _bn_node_key(bn: DiscreteBayesianNetwork, variable: Any) -> Any:
        if variable in bn.nodes():
            return variable
        variable_as_str = str(variable)
        for node in bn.nodes():
            if str(node) == variable_as_str:
                return node
        return variable

    def _source_key_in_source(self, source_var: Any) -> Any:
        if self.source_bn is None:
            return source_var
        return self._bn_node_key(self.source_bn, source_var)

    @staticmethod
    def _cpd_states(cpd: TabularCPD, variable: Any, cardinality: int) -> List[Any]:
        if variable in cpd.state_names:
            return list(cpd.state_names[variable])
        variable_as_str = str(variable)
        if variable_as_str in cpd.state_names:
            return list(cpd.state_names[variable_as_str])
        return list(range(cardinality))

    def transform_to_swig(self, interventions: Dict[str, int]) -> "Swig":
        normalized: Dict[str, int] = {}
        for var, val in interventions.items():
            if var not in self.nodes():
                raise ValueError(f"Intervention impossible : variable '{var}' absente du DAG G.")
            card = self.cardinalities.get(var, 2)
            if not isinstance(val, (int, np.integer)) or not (0 <= int(val) < card):
                raise ValueError(
                    f"La valeur d'intervention pour '{var}' doit etre un entier "
                    f"dans [0, {card - 1}], recu {val!r}."
                )
            normalized[var] = int(val)
        return Swig.from_bn(self, normalized, copy_cpds=True)

    def random_node_for(self, source_var: Any) -> str:
        return self.spec.random_of[str(source_var)]

    def fixed_node_for(self, source_var: Any) -> str:
        return self.spec.fixed_of[str(source_var)]

    def source_variable(self, swig_node: Any) -> str:
        return self.spec.source_of[str(swig_node)]

    def node_info(self, swig_node: Any) -> SwigNodeInfo:
        return self.spec.node_info[str(swig_node)]

    def is_swig_fixed_node(self, swig_node: Any) -> bool:
        return self.spec.node_info[str(swig_node)].is_fixed

    def is_swig_random_node(self, swig_node: Any) -> bool:
        return self.spec.node_info[str(swig_node)].is_random

    def counterfactual_nodes(self) -> List[str]:
        return sorted(
            label for label, info in self.spec.node_info.items()
            if info.is_random and info.fixed_ancestors
        )

    def query_probability(
        self, target: str, state: Any = 1, evidence: Optional[Dict[str, int]] = None
    ) -> float:
        if target not in self.nodes():
            raise ValueError(f"La variable cible '{target}' n'est pas resoluble.")
        inf = VariableElimination(self)
        result = inf.query(variables=[target], evidence=evidence, show_progress=False)
        target_states = list(result.state_names[target])
        if state not in target_states:
            raise ValueError(
                f"state {state!r} not in target states {target_states!r} for variable {target!r}"
            )
        return float(round(result.values[target_states.index(state)], 6))

    def check_d_separation(
        self,
        target: Any, treatment: Any,
        observed_vars: Optional[List] = None,
        forbid_latents: bool = True,
    ) -> bool:
        observed = list(observed_vars or [])
        if forbid_latents:
            for obs in observed:
                base = _get_base_variable(str(obs))
                if base in self.latents:
                    raise ValueError(
                        f"Rupture d'observabilite : impossible de conditionner "
                        f"l'ancetre latent '{obs}'."
                    )
        return self.counterfactual_d_separation(
            self, {treatment}, {target}, set(observed)
        )

    @staticmethod
    def counterfactual_d_separation(
        graph: nx.DiGraph, X: Set, Y: Set, Z: Set
    ) -> bool:
        X_set, Y_set, Z_set = set(X), set(Y), set(Z)
        random_vars, _ = _partition_nodes(graph)
        for label, subset in (("X", X_set), ("Y", Y_set), ("Z", Z_set)):
            invalid = subset - random_vars
            if invalid:
                raise ValueError(
                    f"D-separation : l'ensemble {label} contient des elements "
                    f"non aleatoires : {invalid}"
                )
        purified = graph.subgraph(random_vars)
        return _is_d_separator(purified, X_set, Y_set, Z_set)

    def swig_extended_g_computation(
        self, graph_G: nx.DiGraph, graph_swig: nx.DiGraph
    ) -> str:
        hidden = {
            n for n in graph_G.nodes()
            if graph_G.nodes[n].get("hidden", False)
            or graph_G.nodes[n].get("observed", True) is False
            or graph_G.nodes[n].get("latent", False)
        }
        if hidden:
            raise ValueError(f"G-computation : variables cachees detectees : {hidden}")
        formula = " * ".join(f.expr for f in self._build_g_factors(graph_G, graph_swig))
        for tok in ("fixed", "concrete_fixed", "frozenset", "tuple"):
            if tok in formula:
                raise ValueError(
                    f"G-computation : token interne '{tok}' a fui dans l'expression."
                )
        return formula

    def _build_g_factors(self, graph_G: nx.DiGraph, graph_swig: nx.DiGraph):
        class _Factor:
            __slots__ = ("expr", "vars")
            def __init__(self, expr, vars_):
                self.expr = expr
                self.vars = vars_

        random_vars, fixed_nodes = _partition_nodes(graph_swig)
        factors: List[_Factor] = []
        intervention_registry: Dict[str, str] = {}
        for v in sorted(random_vars, key=lambda x: str(x)):
            base = _get_base_variable(str(v))
            target_g = next(
                (n for n in graph_G.nodes() if n == base or str(n) == str(base)), None
            )
            if target_g is None:
                raise ValueError(f"G-computation : variable '{base}' absente de G.")
            v_sym = str(target_g)
            swig_parents = list(graph_swig.predecessors(v))
            dep: Set[str] = {v_sym}
            if not swig_parents:
                factors.append(_Factor(expr=f"P({v_sym})", vars_=dep))
                continue
            cond_elements = []
            for p in swig_parents:
                p_base = str(_get_base_variable(str(p)))
                dep.add(p_base)
                if p in fixed_nodes:
                    p_str = str(p)
                    if "=" in p_str:
                        p_val = p_str.split("=", 1)[1].rstrip(")")
                    else:
                        p_val = f"a_{p_base}"
                    term = f"{p_base}={p_val}"
                    if p_base in intervention_registry and intervention_registry[p_base] != p_val:
                        raise ValueError(
                            f"G-computation : fluctuation d'intervention pour '{p_base}'."
                        )
                    intervention_registry[p_base] = p_val
                else:
                    term = p_base
                cond_elements.append((p_base, term))
            cond_elements.sort(key=lambda x: x[0])
            cond_str = ", ".join(t[1] for t in cond_elements)
            factors.append(_Factor(expr=f"P({v_sym} | {cond_str})", vars_=dep))
        return factors

    def swig_counterfactual_adjustment_criterion(
        self, graph_swig: nx.DiGraph, X: Any, Y_counterfactual: Any, L: Set
    ) -> Tuple[bool, str]:
        L_set = set(L)
        random_vars, fixed_nodes = _partition_nodes(graph_swig)
        if X not in random_vars:
            raise ValueError(f"Adjustment : X={X} doit etre une variable aleatoire.")
        if Y_counterfactual not in random_vars:
            raise ValueError(
                f"Adjustment : Y={Y_counterfactual} doit etre une variable aleatoire."
            )
        if not L_set.issubset(random_vars):
            raise ValueError("Adjustment : L doit etre inclus dans les variables aleatoires.")
        for l_val in L_set:
            if _is_counterfactual_node(str(l_val)) and ")" in str(l_val):
                parts = str(l_val).split("(", 1)
                if len(parts) > 1 and parts[1].rstrip(")").strip():
                    raise ValueError(
                        "Adjustment : L contient un contrefactuel post-traitement interdit."
                    )
        active_interventions: Dict[str, str] = {}
        for n in fixed_nodes:
            n_str = str(n)
            if "=" in n_str:
                inner = n_str[3:-1]
                parts = inner.split("=", 1)
                active_interventions[parts[0]] = parts[1]
            else:
                active_interventions[n_str] = f"a_{n_str}"
        if str(X) not in active_interventions:
            raise ValueError(f"Adjustment : '{X}' n'a pas de composant fixe actif.")
        A_set = {
            v for v in random_vars if str(_get_base_variable(str(v))) in active_interventions
        }
        is_identifiable = self.counterfactual_d_separation(
            graph_swig, A_set, {Y_counterfactual}, L_set
        )
        if not is_identifiable:
            return (False, "")
        Y_base = str(_get_base_variable(str(Y_counterfactual)))
        sorted_L = sorted(str(_get_base_variable(str(l))) for l in L_set)
        y_cond = list(sorted_L)
        for inv_var, inv_val in sorted(active_interventions.items()):
            y_cond.append(f"{inv_var}={inv_val}")
        y_cond_str = ", ".join(y_cond)
        if sorted_L:
            formula = (
                f"Sum_{{{', '.join(sorted_L)}}} "
                f"P({Y_base} | {y_cond_str}) * P({', '.join(sorted_L)})"
            )
        else:
            formula = f"P({Y_base} | {y_cond_str})"
        return (True, formula)

    def swig_recursive_counterfactual_reduction(
        self, graph_G: nx.DiGraph, V: Any, R: Set, r_tilde: Dict[Any, Any],
        _path: Optional[Set] = None,
    ) -> str:
        if _path is None:
            _path = set()
        if V in _path:
            raise ValueError(f"Reduction : cycle ou boucle infinie detectee a '{V}'.")
        current_path = _path | {V}
        target_g = None
        for n in graph_G.nodes():
            if n == V or str(n) == str(V):
                target_g = n
                break
        if target_g is None:
            raise ValueError(f"Reduction : variable '{V}' absente de G.")
        V_str = str(target_g)
        R_set = set(R)
        for r_node in R_set:
            if r_node == target_g or str(r_node) == V_str:
                return str(r_tilde.get(r_node, f"a_{V_str}"))
        pa_G = set(graph_G.predecessors(target_g))
        if pa_G.issubset(R_set):
            relevant = pa_G & R_set
            if not relevant:
                return V_str
            assignments = []
            for p in sorted(relevant, key=lambda x: str(x)):
                p_str = str(p)
                p_val = str(r_tilde.get(p, f"a_{p_str}"))
                assignments.append(f"{p_str}={p_val}")
            return f"{V_str}({', '.join(assignments)})"
        index_elements: List[str] = []
        intervened_parents = pa_G & R_set
        random_parents = pa_G - R_set
        for p in sorted(intervened_parents, key=lambda x: str(x)):
            p_val = str(r_tilde.get(p, f"a_{p}"))
            index_elements.append(f"{p}={p_val}")
        for p in sorted(random_parents, key=lambda x: str(x)):
            expr = self.swig_recursive_counterfactual_reduction(
                graph_G, p, R_set, r_tilde, current_path
            )
            index_elements.append(expr)
        index_elements.sort(key=lambda x: x.split("(")[0].split("=")[0])
        return f"{V_str}({', '.join(index_elements)})"

    def swig_marginal_g_computation(
        self, graph_G: nx.DiGraph, graph_swig: nx.DiGraph, Y: Set, A: Set
    ) -> str:
        class _Factor:
            __slots__ = ("expr", "vars")
            def __init__(self, expr, vars_):
                self.expr = expr
                self.vars = vars_

        V_strs = {str(v) for v in graph_G.nodes()}
        Y_strs = {str(y) for y in Y}
        A_strs = {str(a) for a in A}
        if not A_strs.issubset(V_strs):
            raise ValueError(f"Marginalisation : A not subset V. Inconnus : {A_strs - V_strs}")
        if not Y_strs.issubset(V_strs):
            raise ValueError(f"Marginalisation : Y not subset V. Inconnus : {Y_strs - V_strs}")
        if A_strs & Y_strs:
            raise ValueError(
                f"Marginalisation : A et Y doivent etre strictement disjoints : {A_strs & Y_strs}"
            )
        W_strs = V_strs - (A_strs | Y_strs)
        factors = self._build_g_factors(graph_G, graph_swig)
        for w in sorted(W_strs):
            dep_factors = [f for f in factors if w in f.vars]
            indep_factors = [f for f in factors if w not in f.vars]
            if dep_factors:
                dep_factors.sort(key=lambda x: x.expr)
                combined = " * ".join(f.expr for f in dep_factors)
                pushed = f"Sum_{{{w}}} [{combined}]"
                merged_vars = set().union(*(f.vars for f in dep_factors)) - {w}
                factors = indep_factors + [_Factor(expr=pushed, vars_=merged_vars)]
        factors.sort(key=lambda x: x.expr)
        return " * ".join(f.expr for f in factors)

    def swig_ett_identification(
        self,
        graph_G: nx.DiGraph,
        graph_swig_x1: nx.DiGraph,
        graph_swig_x0: nx.DiGraph,
        X: Any, Y: Any,
        x_1: str, x_0: str,
    ) -> str:
        if graph_swig_x1 is graph_swig_x0:
            raise ValueError("ETT : les deux SWIGs doivent provenir d'instances distinctes.")

        def _find_random(g: nx.DiGraph, base: Any) -> Optional[Any]:
            for n in g.nodes():
                if _is_fixed_node(str(n)):
                    continue
                b = str(_get_base_variable(str(n)))
                if b == base or b == str(base):
                    return n
            return None

        X1 = _find_random(graph_swig_x1, X)
        Y1 = _find_random(graph_swig_x1, Y)
        X0 = _find_random(graph_swig_x0, X)
        Y0 = _find_random(graph_swig_x0, Y)
        if not all([X1, Y1, X0, Y0]):
            raise ValueError("ETT : variables d'analyse critiques manquantes dans les mondes SWIG.")
        first = f"E[{Y} | X={x_1}]"
        is_unconf = self.counterfactual_d_separation(graph_swig_x0, {X0}, {Y0}, set())
        if is_unconf:
            second = f"E[{Y} | X={x_0}]"
        else:
            parents_X = set(graph_G.predecessors(X))
            L0: Set = set()
            for p in parents_X:
                resolved = _find_random(graph_swig_x0, p)
                if resolved is None:
                    raise ValueError(f"ETT : parent direct '{p}' de X non resolu dans G(x_0).")
                if _is_counterfactual_node(str(resolved)):
                    parts = str(resolved).split("(", 1)
                    if len(parts) > 1 and parts[1].rstrip(")").strip():
                        raise ValueError(
                            f"ETT : parent '{p}' corrompu par un indice contrefactuel en aval."
                        )
                L0.add(resolved)
            if L0 and self.counterfactual_d_separation(graph_swig_x0, {X0}, {Y0}, L0):
                sorted_L = sorted(str(_get_base_variable(str(l))) for l in L0)
                second = (
                    f"Sum_{{{', '.join(sorted_L)}}} "
                    f"E[{Y} | {', '.join(sorted_L)}, X={x_0}] * "
                    f"P({', '.join(sorted_L)} | X={x_1})"
                )
            else:
                raise ValueError("ETT : structure non-identifiable, backdoor non bloquee.")
        return f"{first} - {second}"

    def swig_sequential_plan_evaluation_criterion(
        self,
        graph_swig: nx.DiGraph,
        ordered_treatments: List,
        ordered_covariates: List[Set],
        Y_counterfactual: Any,
    ) -> bool:
        k = len(ordered_treatments)
        if len(ordered_covariates) != k:
            raise ValueError(
                f"Plan sequentiel : {k} etapes mais {len(ordered_covariates)} ensembles de covariables."
            )
        random_vars, fixed_nodes = _partition_nodes(graph_swig)
        if Y_counterfactual not in random_vars:
            raise ValueError(f"Plan seq. : cible Y={Y_counterfactual} doit etre probabiliste.")
        for m in range(k):
            X_m = ordered_treatments[m]
            L_m = set(ordered_covariates[m])
            if X_m not in random_vars:
                raise ValueError(f"Plan seq. : traitement '{X_m}' non aleatoire.")
            if not L_m.issubset(random_vars):
                raise ValueError(f"Plan seq. : covariables L_{m} non aleatoires.")
            if not any(str(_get_base_variable(str(n))) == str(X_m) for n in fixed_nodes):
                raise ValueError(f"Plan seq. : le traitement '{X_m}' n'a pas ete scinde.")
        accumulated_cov: Set = set()
        accumulated_treat: Set = set()
        all_nodes = set(graph_swig.nodes())
        for m in range(k):
            X_m = ordered_treatments[m]
            L_m = set(ordered_covariates[m])
            accumulated_cov.update(L_m)
            if m > 0:
                accumulated_treat.add(ordered_treatments[m - 1])
            H_m = accumulated_cov | accumulated_treat
            future_treats = {str(ordered_treatments[j]) for j in range(m + 1, k)}
            future_covs = set()
            for j in range(m + 1, k):
                future_covs.update(str(c) for c in ordered_covariates[j])
            horizon = future_treats | future_covs
            to_purge = set()
            for node in all_nodes:
                base = str(_get_base_variable(str(node)))
                if base in horizon:
                    to_purge.add(node)
                    continue
                node_str = str(node)
                if _is_counterfactual_node(node_str):
                    inner = node_str.split("(", 1)[1].rstrip(")")
                    for item in inner.split(","):
                        item_base = item.split("=", 1)[0].strip()
                        if item_base in future_treats:
                            to_purge.add(node)
                            break
            subgraph = graph_swig.subgraph(all_nodes - to_purge)
            if not self.counterfactual_d_separation(subgraph, {X_m}, {Y_counterfactual}, H_m):
                return False
        return True

    def swig_joint_factorization(self, graph: nx.DiGraph) -> str:
        random_vars, _ = _partition_nodes(graph)
        terms: List[str] = []
        for v in sorted(random_vars, key=lambda x: str(x)):
            parents = list(graph.predecessors(v))
            v_sym = _node_to_str(v)
            parent_syms = sorted(_node_to_str(p) for p in parents)
            if parent_syms:
                terms.append(f"P({v_sym} | {', '.join(parent_syms)})")
            else:
                terms.append(f"P({v_sym})")
        return " * ".join(terms)

    def render_graph_description(self) -> str:
        return " ".join(
            f"Node {n} points to {', '.join(map(str, sorted(self.successors(n))))}."
            if self.out_degree(n) > 0
            else f"Node {n} has no outgoing links."
            for n in sorted(self.nodes())
        )

    def cpds_text(self) -> str:
        cpds_by_var = {str(cpd.variable): cpd for cpd in self.get_cpds()}
        return "\n\n".join(self._cpd_to_text(cpds_by_var[n]) for n in sorted(cpds_by_var.keys()))

    @staticmethod
    def _cpd_to_text(cpd: TabularCPD, n_round: int = 6) -> str:
        child = str(cpd.variable)
        child_states = Swig._cpd_states(cpd, cpd.variable, int(cpd.variable_card))
        parent_vars = list(cpd.variables[1:])
        parent_cards = [int(c) for c in cpd.cardinality[1:]]
        parent_states_list = [
            Swig._cpd_states(cpd, p, c) for p, c in zip(parent_vars, parent_cards)
        ]
        values = cpd.get_values()
        if not parent_vars:
            return "\n".join(
                f"P({child}={state}) = {round(float(values[i, 0]), n_round)}"
                for i, state in enumerate(child_states)
            )
        lines: List[str] = []
        for col, cfg in enumerate(product(*parent_states_list)):
            cond_str = ", ".join(f"{str(p)}={v}" for p, v in zip(parent_vars, cfg))
            for i, state in enumerate(child_states):
                p = round(float(values[i, col]), n_round)
                lines.append(f"P({child}={state} | {cond_str}) = {p}")
        return "\n".join(lines)

    def validate_swig_metadata(self) -> None:
        graph_nodes = set(map(str, self.nodes()))
        metadata_nodes = set(self.spec.node_info)
        if graph_nodes != metadata_nodes:
            raise ValueError(
                "SWIG metadata/node mismatch: "
                f"graph_only={sorted(graph_nodes - metadata_nodes)}, "
                f"metadata_only={sorted(metadata_nodes - graph_nodes)}"
            )
        for source_var in self.spec.source_nodes:
            if source_var not in self.spec.random_of:
                raise ValueError(f"Missing random node for source variable {source_var}")
        for source_var in self.spec.interventions:
            if source_var not in self.spec.fixed_of:
                raise ValueError(f"Missing fixed node for intervention on {source_var}")

    def to_nl(self) -> str:
        lines = ["Single-World Intervention Graph."]
        if self.spec.interventions:
            intervention_text = ", ".join(
                f"{var} set to {value!r}"
                for var, value in sorted(self.spec.interventions.items())
            )
            lines.append(f"Interventions: {intervention_text}.")
        else:
            lines.append("Interventions: none.")
        fixed = [info for info in self.spec.node_info.values() if info.kind == "fixed"]
        random_nodes = [info for info in self.spec.node_info.values() if info.kind == "random"]
        if fixed:
            lines.append("Fixed nodes:")
            for info in sorted(fixed, key=lambda x: x.label):
                lines.append(
                    f"- {info.label}: fixed value for source variable {info.source}."
                )
        lines.append("Random nodes:")
        for info in sorted(random_nodes, key=lambda x: x.label):
            if info.fixed_ancestors:
                ancestors = ", ".join(info.fixed_ancestors)
                lines.append(
                    f"- {info.label}: source variable {info.source} under "
                    f"fixed ancestors {ancestors}."
                )
            else:
                lines.append(f"- {info.label}: source variable {info.source}.")
        lines.append("Edges:")
        for parent, child in sorted(self.edges()):
            lines.append(f"- {parent} -> {child}")
        return "\n".join(lines)

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "nodes": list(self.nodes()),
            "edges": list(self.edges()),
            "trace": list(self.trace),
        }

    @classmethod
    def from_serializable(
        cls, data: Mapping[str, Any], *, source_bn: Optional["Swig"] = None
    ) -> "Swig":
        spec = SwigSpec.from_dict(data["spec"])
        swig = cls(source_bn=source_bn, spec=spec)
        swig.add_nodes_from(data.get("nodes", ()))
        swig.add_edges_from(data.get("edges", ()))
        swig.trace = list(data.get("trace", ()))
        if swig.source_bn is not None:
            swig._build_swig_cpds()
        return swig

    def copy(self) -> "Swig":
        copied = Swig(
            config=copy.deepcopy(self.config),
            source_bn=self.source_bn,
            spec=SwigSpec.from_dict(self.spec.to_dict()),
        )
        copied.add_nodes_from(self.nodes())
        copied.add_edges_from(self.edges())
        copied.trace = list(self.trace)
        copied.cardinalities = dict(self.cardinalities)
        copied.latents = set(self.latents)
        for cpd in self.get_cpds():
            copied.add_cpds(cpd.copy())
        return copied

class SwigCounterfactualEngine:
    """
    Moteur d'inference contrefactuelle exacte par enumeration des fonctions
    de reponse (SCM canonique Markovien).

    Algorithme en 3 etapes :
    1. Abduction  : conditionne les tables de reponse sur l'evidence factuelle
    2. Action     : applique l'intervention (do-operator) dans le SWIG
    3. Prediction : lit la variable cible dans le monde contrefactuel
    """

    def __init__(self, swig: Swig, *, max_response_functions: int = 200_000):
        if swig.source_bn is None:
            raise ValueError("Counterfactual inference requires swig.source_bn.")
        if swig.source_bn.latents or swig.latents:
            raise ValueError(
                "SwigCounterfactualEngine assumes a Markovian (latent-free) SCM; "
                "the source network contains latent confounders."
            )
        self.swig = swig
        self.max_response_functions = max_response_functions
        self.mechanisms = self._build_mechanisms()
        self.topological_order = self._source_topological_order()
        self.trace: List[str] = []
        self.last_evidence_probability: Optional[float] = None

    def response_space_size(self, *, positive_only: bool = True) -> int:
        size = 1
        for mechanism in self.mechanisms.values():
            for config in mechanism.parent_configs:
                probs = mechanism.probabilities[config]
                if positive_only:
                    n_choices = sum(prob > 0 for prob in probs)
                else:
                    n_choices = len(probs)
                size *= n_choices
                if size > self.max_response_functions:
                    return size
        return size

    def explain_query(
        self,
        target: Any,
        factual_evidence: Optional[Mapping[Any, Any]] = None,
    ) -> str:
        target_source = self._target_source(target)
        target_node = self.swig.random_node_for(target_source)
        evidence = self._normalize_factual_evidence(factual_evidence or {})
        response_space = self.response_space_size()
        self.trace = [
            f"Goal: compute distribution for {target_node}.",
            "Semantics: canonical discrete response-table SCM induced by the source BN CPDs.",
            f"Response functions to enumerate: {response_space}.",
            f"Abduction: condition response tables on factual evidence {evidence}.",
            f"Action: apply interventions {self.swig.spec.interventions}.",
            f"Prediction: read source variable {target_source} in the intervened world.",
        ]
        return "\n".join(self.trace)

    def query(
        self,
        target: Any,
        factual_evidence: Optional[Mapping[Any, Any]] = None,
        *,
        n_round: Optional[int] = None,
    ) -> Dict[Any, float]:
        target_source = self._target_source(target)
        evidence = self._normalize_factual_evidence(factual_evidence or {})
        response_space = self.response_space_size()
        if response_space > self.max_response_functions:
            raise RuntimeError(
                "Exact response-function enumeration would visit "
                f"{response_space} functions, above max_response_functions="
                f"{self.max_response_functions}."
            )
        target_states = self.mechanisms[target_source].states
        weights = {state: 0.0 for state in target_states}
        evidence_weight = 0.0
        visited = 0

        for response_tables, response_weight in self._iter_response_tables():
            visited += 1
            factual_world = self._evaluate_world(response_tables, interventions={})
            if not self._world_matches(factual_world, evidence):
                continue
            evidence_weight += response_weight
            counterfactual_world = self._evaluate_world(
                response_tables, interventions=self.swig.spec.interventions,
            )
            weights[counterfactual_world[target_source]] += response_weight

        if evidence_weight <= 0:
            raise ValueError(
                "The factual evidence has zero probability under the canonical "
                f"SCM: {evidence}."
            )
        distribution = {state: weight / evidence_weight for state, weight in weights.items()}
        if n_round is not None:
            distribution = {state: round(prob, n_round) for state, prob in distribution.items()}
        self.last_evidence_probability = evidence_weight
        target_node = self.swig.random_node_for(target_source)
        self.trace = [
            f"Goal: compute distribution for {target_node}.",
            "Semantics: canonical discrete response-table SCM induced by the source BN CPDs.",
            f"Enumerated response functions: {visited}.",
            f"Abduction evidence: {evidence}.",
            f"P(evidence) = {evidence_weight}.",
            f"Action interventions: {self.swig.spec.interventions}.",
            f"Result: {distribution}.",
        ]
        return distribution

    def probability_of_factual_evidence(self, factual_evidence: Mapping[Any, Any]) -> float:
        evidence = self._normalize_factual_evidence(factual_evidence)
        response_space = self.response_space_size()
        if response_space > self.max_response_functions:
            raise RuntimeError(
                "Exact response-function enumeration would visit "
                f"{response_space} functions, above max_response_functions="
                f"{self.max_response_functions}."
            )
        probability = 0.0
        for response_tables, response_weight in self._iter_response_tables():
            factual_world = self._evaluate_world(response_tables, interventions={})
            if self._world_matches(factual_world, evidence):
                probability += response_weight
        return probability

    def _build_mechanisms(self) -> Dict[str, _ResponseMechanism]:
        mechanisms = {}
        for source in self.swig.spec.source_nodes:
            source_key = self.swig._source_key_in_source(source)
            cpd = self.swig.source_bn.get_cpds(source_key)
            if cpd is None:
                raise ValueError(f"Missing CPD for source variable {source!r}.")
            if not isinstance(cpd, TabularCPD):
                raise TypeError(
                    "SwigCounterfactualEngine currently supports TabularCPD "
                    f"only; got {type(cpd)!r} for {source!r}."
                )
            parent_keys = tuple(cpd.variables[1:])
            parents = tuple(str(parent) for parent in parent_keys)
            states = tuple(Swig._cpd_states(cpd, cpd.variable, int(cpd.variable_card)))
            parent_states = tuple(
                tuple(Swig._cpd_states(cpd, parent, int(cardinality)))
                for parent, cardinality in zip(parent_keys, cpd.cardinality[1:])
            )
            parent_configs = tuple(product(*parent_states)) if parent_states else ((),)
            probabilities = {}
            for config in parent_configs:
                probs = tuple(
                    self._cpd_probability(cpd, state, parent_keys, config)
                    for state in states
                )
                total = sum(probs)
                if total <= 0:
                    raise ValueError(
                        f"CPD for {source!r} assigns zero mass to parent "
                        f"configuration {config!r}."
                    )
                if not np.isclose(total, 1.0):
                    probs = tuple(prob / total for prob in probs)
                probabilities[config] = probs
            mechanisms[source] = _ResponseMechanism(
                source=source, source_key=source_key, cpd=cpd, parents=parents,
                parent_keys=parent_keys, states=states, parent_states=parent_states,
                parent_configs=parent_configs, probabilities=probabilities,
            )
        return mechanisms

    def _source_topological_order(self) -> List[str]:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.swig.spec.source_nodes)
        graph.add_edges_from(self.swig.spec.source_edges)
        return list(nx.topological_sort(graph))

    def _iter_response_tables(self):
        sources = list(self.topological_order)

        def rec(index, current_tables, current_weight):
            if index == len(sources):
                yield current_tables.copy(), current_weight
                return
            source = sources[index]
            mechanism = self.mechanisms[source]
            for table, table_weight in self._iter_mechanism_tables(mechanism):
                current_tables[source] = table
                yield from rec(index + 1, current_tables, current_weight * table_weight)
                del current_tables[source]

        yield from rec(0, {}, 1.0)

    def _iter_mechanism_tables(self, mechanism: _ResponseMechanism):
        choices_by_config = []
        for config in mechanism.parent_configs:
            choices = [
                (state, probability)
                for state, probability in zip(mechanism.states, mechanism.probabilities[config])
                if probability > 0
            ]
            choices_by_config.append((config, choices))
        for selected_choices in product(*(choices for _, choices in choices_by_config)):
            table = {}
            weight = 1.0
            for (config, _), (state, probability) in zip(choices_by_config, selected_choices):
                table[config] = state
                weight *= probability
            if weight > 0:
                yield table, weight

    def _evaluate_world(
        self,
        response_tables: Mapping[str, Mapping[Tuple[Any, ...], Any]],
        *,
        interventions: Mapping[str, Any],
    ) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        normalized_interventions = {str(source): value for source, value in interventions.items()}
        for source in self.topological_order:
            if source in normalized_interventions:
                values[source] = normalized_interventions[source]
                continue
            mechanism = self.mechanisms[source]
            parent_config = tuple(values[parent] for parent in mechanism.parents)
            values[source] = response_tables[source][parent_config]
        return values

    @staticmethod
    def _world_matches(
        world: Mapping[str, Any], evidence: Mapping[str, Any]
    ) -> bool:
        return all(world[source] == value for source, value in evidence.items())

    def _normalize_factual_evidence(
        self, evidence: Mapping[Any, Any]
    ) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for raw_key, value in evidence.items():
            key = str(raw_key)
            if key in self.swig.spec.source_nodes:
                source = key
            elif key in self.swig.spec.node_info:
                info = self.swig.spec.node_info[key]
                if info.is_fixed:
                    raise ValueError(
                        "Factual evidence should use source/random variables, "
                        f"not fixed intervention node {key!r}."
                    )
                if info.fixed_ancestors:
                    raise ValueError(
                        "This engine conditions on factual-world evidence only; "
                        f"{key!r} is an intervention-indexed SWIG node."
                    )
                source = info.source
            else:
                raise ValueError(f"Unknown factual evidence variable {raw_key!r}.")
            normalized[source] = value
        return normalized

    def _target_source(self, target: Any) -> str:
        key = str(target)
        if key in self.swig.spec.source_nodes:
            return key
        if key in self.swig.spec.node_info:
            info = self.swig.spec.node_info[key]
            if info.is_fixed:
                raise ValueError(f"Cannot query fixed intervention node {key!r}.")
            return info.source
        raise ValueError(f"Unknown SWIG target node {target!r}.")

    @staticmethod
    def _cpd_probability(
        cpd: TabularCPD,
        state: Any,
        parent_keys: Tuple[Any, ...],
        parent_config: Tuple[Any, ...],
    ) -> float:
        kwargs = {str(cpd.variable): state}
        kwargs.update(
            {str(parent): value for parent, value in zip(parent_keys, parent_config)}
        )
        try:
            return float(cpd.get_value(**kwargs))
        except Exception:
            return SwigCounterfactualEngine._cpd_probability_from_table(
                cpd, state, parent_keys, parent_config,
            )

    @staticmethod
    def _cpd_probability_from_table(
        cpd: TabularCPD,
        state: Any,
        parent_keys: Tuple[Any, ...],
        parent_config: Tuple[Any, ...],
    ) -> float:
        child_states = Swig._cpd_states(cpd, cpd.variable, int(cpd.variable_card))
        child_index = child_states.index(state)
        column_index = 0
        for parent, parent_value, cardinality in zip(
            parent_keys, parent_config, cpd.cardinality[1:]
        ):
            parent_states = Swig._cpd_states(cpd, parent, int(cardinality))
            column_index = column_index * len(parent_states)
            column_index += parent_states.index(parent_value)
        return float(cpd.get_values()[child_index, column_index])


__all__ = [
    "Swig",
    "SwigConfig",
    "SwigSpec",
    "SwigNodeInfo",
    "SwigCounterfactualEngine",
]
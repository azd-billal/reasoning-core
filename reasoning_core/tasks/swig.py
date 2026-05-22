from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import random
import re
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from tasks._causal_utils import get_random_DAG
from reasoning_core.template import Task, Problem, Config

@dataclass
class SwigConfig(Config):
    num_nodes: int = 5
    graph_density: float = 0.4
    num_latents: int = 1
    cardinality: int = 2
    cpd_low: float = 0.1
    cpd_high: float = 0.9
    intervention_value: int = 1
    random_seed: Optional[int] = None

    def update(self, c: float) -> None:
        self.num_nodes = max(2, int(self.num_nodes * (1 + c)))
        self.graph_density = min(0.8, self.graph_density + 0.1 * c)

_MEDICAL_NODES = [
    "Age", "Tabagisme", "Genetique", "Traitement", "Cancer",
    "Stress", "Obesite", "Alcool", "Antecedents", "Activite",
    "IMC", "Cholesterol", "Inflammation", "Immunite", "Hormones",
]

def draw_swig(swig_graph: nx.DiGraph, swig_bn: Optional[DiscreteBayesianNetwork] = None, latents: Optional[Set[str]] = None, title: str = "SWIG", ax=None):
    """Affichage d'un SWIG avec convention de couleurs standard du framework.

    Si swig_bn est fourni, P(node=1) est calculée via VE et affichée
    directement dans le label de chaque nœud aléatoire.
    """
    latents = latents or set()

    marginals: Dict[Any, float] = {}
    if swig_bn is not None:
        try:
            inf = VariableElimination(swig_bn)
            for node in swig_graph.nodes():
                if _is_fixed_node(node):
                    continue
                bn_name = _node_to_str(node)
                if bn_name in swig_bn.nodes():
                    try:
                        q = inf.query(variables=[bn_name], show_progress=False)
                        marginals[node] = float(round(q.values[1], 3))
                    except Exception:
                        pass
        except Exception:
            pass

    pos = nx.circular_layout(swig_graph)
    labels = {}
    colors = []
    for node in swig_graph.nodes():
        base = str(_get_base_variable(node))
        if _is_fixed_node(node):
            color = "lightgreen"
        elif base in latents:
            color = "lightcoral"
        else:
            color = "lightblue"
        colors.append(color)
        label = _node_to_str(node)
        if node in marginals:
            label += f"\n{marginals[node]:.3f}"
        labels[node] = label

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 10))

    ax.set_title(title, fontsize=14, fontweight="bold")
    nx.draw(swig_graph, pos, ax=ax, labels=labels, with_labels=True,
            node_color=colors, node_size=3000, font_size=7,
            font_weight="bold", arrows=True, arrowsize=15)

    if standalone:
        plt.tight_layout()
        plt.show()

def draw_dag_and_swig(swig_model: Swig, swig_graph: nx.DiGraph, swig_bn: Optional[DiscreteBayesianNetwork] = None):
    """Affichage côte à côte du DAG original et du SWIG associé.

    Si swig_bn est fourni, P(node=1) est affiché dans les labels des deux graphes
    (marginale observationnelle pour le DAG, marginale interventionnelle pour le SWIG).
    """
    fig, (ax_dag, ax_swig) = plt.subplots(1, 2, figsize=(18, 8))

    # Marginales observationnelles du DAG original
    dag_marginals: Dict[Any, float] = {}
    try:
        inf = VariableElimination(swig_model)
        for node in swig_model.nodes():
            try:
                q = inf.query(variables=[node], show_progress=False)
                dag_marginals[node] = float(round(q.values[1], 3))
            except Exception:
                pass
    except Exception:
        pass

    pos = nx.circular_layout(swig_model)
    labels = {}
    for n in swig_model.nodes():
        label = str(n)
        if n in dag_marginals:
            label += f"\n{dag_marginals[n]:.3f}"
        labels[n] = label
    colors = ["lightcoral" if n in swig_model.latents else "lightblue" for n in swig_model.nodes()]
    ax_dag.set_title("DAG Original G  —  P(X=1)", fontsize=14, fontweight="bold")
    nx.draw(swig_model, pos, ax=ax_dag, labels=labels, with_labels=True,
            node_color=colors, node_size=3000, font_size=7,
            font_weight="bold", arrows=True, arrowsize=15)

    # Marginales interventionnelles du SWIG (passées à draw_swig via swig_bn)
    draw_swig(swig_graph, swig_bn=swig_bn, latents=swig_model.latents,
              title="SWIG G(ã)  —  P(X(a)=1)", ax=ax_swig)

    plt.tight_layout()
    plt.show()


def draw_dag_and_two_swigs(
    swig_model: "Swig",
    swig_graph_t1: nx.DiGraph,
    swig_graph_t0: nx.DiGraph,
    swig_bn_t1: Optional[DiscreteBayesianNetwork] = None,
    swig_bn_t0: Optional[DiscreteBayesianNetwork] = None,
):
    """Affichage en trois panneaux : DAG original | SWIG(do=1) | SWIG(do=0).

    Utilisé quand une tâche implique deux mondes contrefactuels distincts,
    typiquement pour calculer la Probabilité de Nécessité.
    """
    fig, (ax_dag, ax_t1, ax_t0) = plt.subplots(1, 3, figsize=(27, 8))

    dag_marginals: Dict[Any, float] = {}
    try:
        inf = VariableElimination(swig_model)
        for node in swig_model.nodes():
            try:
                q = inf.query(variables=[node], show_progress=False)
                dag_marginals[node] = float(round(q.values[1], 3))
            except Exception:
                pass
    except Exception:
        pass

    pos = nx.circular_layout(swig_model)
    labels = {
        n: (str(n) + f"\n{dag_marginals[n]:.3f}" if n in dag_marginals else str(n))
        for n in swig_model.nodes()
    }
    colors = ["lightcoral" if n in swig_model.latents else "lightblue" for n in swig_model.nodes()]
    ax_dag.set_title("DAG Original G  —  P(X=1)", fontsize=14, fontweight="bold")
    nx.draw(swig_model, pos, ax=ax_dag, labels=labels, with_labels=True,
            node_color=colors, node_size=3000, font_size=7,
            font_weight="bold", arrows=True, arrowsize=15)

    draw_swig(swig_graph_t1, swig_bn=swig_bn_t1, latents=swig_model.latents,
              title="SWIG G(do=1)  —  P(X(1)=1)", ax=ax_t1)
    draw_swig(swig_graph_t0, swig_bn=swig_bn_t0, latents=swig_model.latents,
              title="SWIG G(do=0)  —  P(X(0)=1)", ax=ax_t0)

    plt.tight_layout()
    plt.show()


def _is_fixed_node(n: Any) -> bool:
    return isinstance(n, tuple) and len(n) >= 2 and n[0] in ("fixed", "concrete_fixed")

def _is_counterfactual_node(n: Any) -> bool:
    return isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], frozenset)

def _get_base_variable(n: Any) -> Any:
    if isinstance(n, tuple):
        if n[0] in ("fixed", "concrete_fixed"):
            return n[1]
        return n[0]
    return n

def _node_to_str(node: Any) -> str:
    if isinstance(node, tuple):
        if node[0] == "fixed":
            return f"a_{node[1]}"
        if node[0] == "concrete_fixed":
            return f"{node[1]}={node[2]}"
        if _is_counterfactual_node(node):
            V, a_V = node
            items = []
            for item in sorted(a_V, key=lambda x: str(x)):
                if isinstance(item, tuple) and len(item) == 2:
                    items.append(f"a_{item[1]}" if item[0] == "fixed" else f"{item[0]}={item[1]}")
                else:
                    items.append(str(item))
            return f"{V}({','.join(items)})"
    return str(node)

@dataclass
class _Factor:
    expr: str
    vars: Set[str]

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

def _partition_nodes(graph: nx.DiGraph) -> Tuple[Set, Set]:
    all_nodes = set(graph.nodes())
    fixed = {n for n in all_nodes if _is_fixed_node(n)}
    return all_nodes - fixed, fixed

if hasattr(nx, "is_d_separator"):
    _nx_d_sep = nx.is_d_separator
elif hasattr(nx, "d_separated"):
    _nx_d_sep = nx.d_separated
else:
    raise ImportError("NetworkX incompatible : aucune fonction de d-séparation trouvée.")

def _is_d_separator(G: nx.DiGraph, X: Set, Y: Set, Z: Set) -> bool:
    return _nx_d_sep(G, X, Y, Z)

class Swig(DiscreteBayesianNetwork):
    """
    Réseau Bayésien Causal étendu supportant le formalisme SWIG (Robins & Richardson).
    Hérite directement de DiscreteBayesianNetwork pour encapsuler l'inférence factuelle.
    """

    def __init__(self, config: Optional[SwigConfig] = None):
        super().__init__()
        self.config = config if config is not None else SwigConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        # Dict parallèle pour éviter de passer par l'API add_cpds de pgmpy avant
        # que la structure du graphe soit complète — add_cpds valide immédiatement.
        self.custom_cpds: Dict[str, TabularCPD] = {}
        self.cardinalities: Dict[str, int] = {}
        # Passe par le setter pgmpy (_GraphRolesMixin.latents) ; l'ensemble vide
        # est valide ici car le graphe est encore vide — aucun nœud à valider.
        self.latents: Set[str] = set()

    def node_split_transform(self, graph: nx.DiGraph, treatments: Iterable) -> nx.DiGraph:
        """SWIG node-splitting transformation (Séparation des nœuds traités)."""
        A = set(treatments)
        V = set(graph.nodes())
        if not A <= V:
            raise ValueError(f"L'ensemble d'intervention A doit être ⊆ V. Hors V : {A - V}")

        fixed = {t: ("fixed", t) for t in A}
        star = nx.DiGraph()
        star.add_nodes_from(V)
        star.add_nodes_from(fixed.values())
        for u, v in graph.edges():
            star.add_edge(fixed[u] if u in A else u, v)

        if not nx.is_directed_acyclic_graph(star):
            raise ValueError("G* n'est pas acyclique.")
        return star

    def counterfactual_labeling(self, star: nx.DiGraph, fixed_nodes: Iterable) -> nx.DiGraph:
        """Étape de labellisation contrefactuelle par détection d'ancêtres fixes."""
        nodes = set(star.nodes())
        fixed_set = set(fixed_nodes)
        if not fixed_set <= nodes:
            raise ValueError(f"fixed_nodes doit ⊆ vertices(G*). Inconnus : {fixed_set - nodes}")

        random_nodes = nodes - fixed_set
        a_index = {V: frozenset(nx.ancestors(star, V) & fixed_set) for V in random_nodes}
        final_label = {n: (n, a_index[n]) if (n in random_nodes and a_index[n]) else n for n in nodes}

        swit = nx.DiGraph()
        swit.add_nodes_from(final_label[n] for n in star.nodes())
        swit.add_edges_from((final_label[u], final_label[v]) for u, v in star.edges())
        return swit

    def swig_instantiation(self, swit: nx.DiGraph, assignments: Dict[Any, Any]) -> nx.DiGraph:
        """Instanciation concrète G(a) -> G(ã) par assignation des valeurs numériques."""
        mapping = {}
        for node in swit.nodes():
            if isinstance(node, tuple) and len(node) == 2 and node[0] == "fixed":
                t = node[1]
                if t not in assignments:
                    raise ValueError(f"Instanciation : valeur manquante pour la variable libre '{t}'.")
                mapping[node] = ("concrete_fixed", t, assignments[t])
            elif _is_counterfactual_node(node):
                V, a_V = node
                pairs = []
                for item in a_V:
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == "fixed":
                        t = item[1]
                        if t not in assignments:
                            raise ValueError(f"Instanciation : valeur libre '{t}' non résolue pour '{V}'.")
                        pairs.append((t, assignments[t]))
                    else:
                        raise ValueError(f"Élément structurel mal formé dans l'index de '{V}' : {item}")
                mapping[node] = (V, frozenset(pairs))
            else:
                mapping[node] = node

        g_concrete = nx.DiGraph()
        g_concrete.add_nodes_from(mapping.values())
        g_concrete.add_edges_from((mapping[u], mapping[v]) for u, v in swit.edges())
        return g_concrete

    def counterfactual_d_separation(self, graph: nx.DiGraph, X: Set, Y: Set, Z: Set) -> bool:
        """D-séparation contrefactuelle restreinte exclusivement aux variables aléatoires."""
        X_set, Y_set, Z_set = set(X), set(Y), set(Z)
        random_vars, _ = _partition_nodes(graph)

        for label, subset in (("X", X_set), ("Y", Y_set), ("Z", Z_set)):
            invalid = subset - random_vars
            if invalid:
                raise ValueError(f"D-séparation : l'ensemble {label} contient des éléments non aléatoires : {invalid}")

        purified = graph.subgraph(random_vars)
        return _is_d_separator(purified, X_set, Y_set, Z_set)

    def swig_joint_factorization(self, graph: nx.DiGraph) -> str:
        """Factorisation jointe contrefactuelle sur les nœuds probabilistes du SWIG."""
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

    def swig_modularity_transformation(self, graph_G: nx.DiGraph, graph_swig: nx.DiGraph, node_counterfactual: Any) -> str:
        """Remplacement modulaire d'un terme contrefactuel par son homologue factuel."""
        if node_counterfactual not in graph_swig:
            raise ValueError(f"Modularité : '{node_counterfactual}' absent du SWIG.")

        _, fixed_nodes = _partition_nodes(graph_swig)
        if node_counterfactual in fixed_nodes:
            raise ValueError("Modularité : le nœud cible ne peut pas être une constante fixe.")

        base = _get_base_variable(node_counterfactual)
        target_g = None
        for node in graph_G.nodes():
            if node == base or str(node) == str(base):
                target_g = node
                break
        if target_g is None:
            raise ValueError(f"Modularité : variable de base '{base}' absente de G.")

        expected_parents = {str(p) for p in graph_G.predecessors(target_g)}
        swig_parents = list(graph_swig.predecessors(node_counterfactual))
        actual_parents = {str(_get_base_variable(p)) for p in swig_parents}
        if expected_parents != actual_parents:
            raise ValueError(f"Modularité : parents incohérents. Attendu {expected_parents}, obtenu {actual_parents}.")

        if not swig_parents:
            return f"P({target_g})"

        cond_terms = []
        for p in swig_parents:
            p_base = str(_get_base_variable(p))
            if p in fixed_nodes:
                if isinstance(p, tuple) and p[0] == "concrete_fixed":
                    cond_terms.append((p_base, f"{p_base}={p[2]}"))
                else:
                    cond_terms.append((p_base, f"{p_base}=a_{p_base}"))
            else:
                cond_terms.append((p_base, p_base))

        cond_terms.sort(key=lambda x: x[0])
        cond_str = ", ".join(t[1] for t in cond_terms)
        return f"P({target_g} | {cond_str})"

    def _build_g_factors(self, graph_G: nx.DiGraph, graph_swig: nx.DiGraph) -> List[_Factor]:
        """Construit la liste structurée des facteurs de la g-formula étendue."""
        random_vars, fixed_nodes = _partition_nodes(graph_swig)
        factors: List[_Factor] = []
        intervention_registry: Dict[str, str] = {}

        for v in sorted(random_vars, key=lambda x: str(x)):
            base = _get_base_variable(v)
            target_g = next((n for n in graph_G.nodes() if n == base or str(n) == str(base)), None)
            if target_g is None:
                raise ValueError(f"G-computation : variable '{base}' absente de G.")

            v_sym = str(target_g)
            swig_parents = list(graph_swig.predecessors(v))
            dep: Set[str] = {v_sym}

            if not swig_parents:
                factors.append(_Factor(expr=f"P({v_sym})", vars=dep))
                continue

            cond_elements = []
            for p in swig_parents:
                p_base = str(_get_base_variable(p))
                dep.add(p_base)
                if p in fixed_nodes:
                    p_val = str(p[2]) if (isinstance(p, tuple) and p[0] == "concrete_fixed") else f"a_{p_base}"
                    term = f"{p_base}={p_val}"
                    if p_base in intervention_registry and intervention_registry[p_base] != p_val:
                        raise ValueError(f"G-computation : fluctuation d'intervention pour '{p_base}'.")
                    intervention_registry[p_base] = p_val
                else:
                    term = p_base
                cond_elements.append((p_base, term))

            cond_elements.sort(key=lambda x: x[0])
            cond_str = ", ".join(t[1] for t in cond_elements)
            factors.append(_Factor(expr=f"P({v_sym} | {cond_str})", vars=dep))

        return factors

    def swig_extended_g_computation(self, graph_G: nx.DiGraph, graph_swig: nx.DiGraph) -> str:
        """Formule globale de g-computation étendue sans fuite de structure contrefactuelle."""
        hidden = {n for n in graph_G.nodes() if graph_G.nodes[n].get('hidden', False) or graph_G.nodes[n].get('observed', True) is False or graph_G.nodes[n].get('latent', False)}
        if hidden:
            raise ValueError(f"G-computation : variables cachées détectées : {hidden}")

        formula = " * ".join(f.expr for f in self._build_g_factors(graph_G, graph_swig))
        for tok in ("fixed", "concrete_fixed", "frozenset", "tuple"):
            if tok in formula:
                raise ValueError(f"G-computation : token interne '{tok}' a fui dans l'expression.")
        return formula

    def swig_counterfactual_adjustment_criterion(self, graph_swig: nx.DiGraph, X: Any, Y_counterfactual: Any, L: Set) -> Tuple[bool, str]:
        """Vérification du critère d'ajustement contrefactuel généralisé aux interventions multiples."""
        L_set = set(L)
        random_vars, fixed_nodes = _partition_nodes(graph_swig)

        if X not in random_vars:
            raise ValueError(f"Adjustment : X={X} doit être une variable aléatoire.")
        if Y_counterfactual not in random_vars:
            raise ValueError(f"Adjustment : Y={Y_counterfactual} doit être une variable aléatoire.")
        if not L_set.issubset(random_vars):
            raise ValueError("Adjustment : L doit ⊆ variables aléatoires.")

        for l in L_set:
            if _is_counterfactual_node(l) and len(l[1]) > 0:
                raise ValueError("Adjustment : L contient un contrefactuel post-traitement interdit.")

        active_interventions: Dict[str, str] = {}
        for n in fixed_nodes:
            if n[0] == "concrete_fixed":
                active_interventions[str(n[1])] = str(n[2])
            elif n[0] == "fixed":
                active_interventions[str(n[1])] = f"a_{n[1]}"

        if str(X) not in active_interventions:
            raise ValueError(f"Adjustment : '{X}' n'a pas de composant fixe actif d'intervention.")

        A_set = {v for v in random_vars if str(_get_base_variable(v)) in active_interventions}
        is_identifiable = self.counterfactual_d_separation(graph_swig, A_set, {Y_counterfactual}, L_set)
        if not is_identifiable:
            return (False, "")

        Y_base = str(_get_base_variable(Y_counterfactual))
        sorted_L = sorted(str(_get_base_variable(l)) for l in L_set)

        y_cond = list(sorted_L)
        for inv_var, inv_val in sorted(active_interventions.items()):
            y_cond.append(f"{inv_var}={inv_val}")
        y_cond_str = ", ".join(y_cond)

        if sorted_L:
            formula = f"Sum_{{{', '.join(sorted_L)}}} P({Y_base} | {y_cond_str}) * P({', '.join(sorted_L)})"
        else:
            formula = f"P({Y_base} | {y_cond_str})"
        return (True, formula)

    def swig_recursive_counterfactual_reduction(self, graph_G: nx.DiGraph, V: Any, R: Set, r_tilde: Dict[Any, Any], _path: Optional[Set] = None) -> str:
        """Réduction contrefactuelle récursive selon l'axiome de consistance et restriction d'exclusion."""
        if _path is None:
            _path = set()
        if V in _path:
            raise ValueError(f"Réduction : cycle ou boucle infinie détectée à '{V}'.")
        current_path = _path | {V}

        target_g = None
        for n in graph_G.nodes():
            if n == V or str(n) == str(V):
                target_g = n
                break
        if target_g is None:
            raise ValueError(f"Réduction : variable '{V}' absente de G.")

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
            expr = self.swig_recursive_counterfactual_reduction(graph_G, p, R_set, r_tilde, current_path)
            index_elements.append(expr)

        index_elements.sort(key=lambda x: x.split("(")[0].split("=")[0])
        return f"{V_str}({', '.join(index_elements)})"

    def swig_marginal_g_computation(self, graph_G: nx.DiGraph, graph_swig: nx.DiGraph, Y: Set, A: Set) -> str:
        """Marginalisation push-sum de la g-formula étendue sur W = V \\ (A ∪ Y)."""
        V_strs = {str(v) for v in graph_G.nodes()}
        Y_strs = {str(y) for y in Y}
        A_strs = {str(a) for a in A}

        if not A_strs.issubset(V_strs):
            raise ValueError(f"Marginalisation : A ⊄ V. Inconnus : {A_strs - V_strs}")
        if not Y_strs.issubset(V_strs):
            raise ValueError(f"Marginalisation : Y ⊄ V. Inconnus : {Y_strs - V_strs}")
        if A_strs & Y_strs:
            raise ValueError(f"Marginalisation : A et Y doivent être strictement disjoints : {A_strs & Y_strs}")

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
                factors = indep_factors + [_Factor(expr=pushed, vars=merged_vars)]

        factors.sort(key=lambda x: x.expr)
        return " * ".join(f.expr for f in factors)

    def swig_ett_identification(self, graph_G: nx.DiGraph, graph_swig_x1: nx.DiGraph, graph_swig_x0: nx.DiGraph, X: Any, Y: Any, x_1: str, x_0: str) -> str:
        """Identification de l'effet du traitement sur les traités (ETT) avec isolation croisée des mondes."""
        if graph_swig_x1 is graph_swig_x0:
            raise ValueError("ETT : les deux SWIGs doivent provenir d'instances de graphes distinctes.")

        def _find_random(g: nx.DiGraph, base: Any) -> Optional[Any]:
            for n in g.nodes():
                if _is_fixed_node(n):
                    continue
                b = _get_base_variable(n)
                if b == base or str(b) == str(base):
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
                    raise ValueError(f"ETT : parent direct '{p}' de X non résolu dans G(x_0).")
                if _is_counterfactual_node(resolved) and len(resolved[1]) > 0:
                    raise ValueError(f"ETT : parent '{p}' corrompu par un indice contrefactuel en aval.")
                L0.add(resolved)
            if L0 and self.counterfactual_d_separation(graph_swig_x0, {X0}, {Y0}, L0):
                sorted_L = sorted(str(_get_base_variable(l)) for l in L0)
                second = f"Sum_{{{', '.join(sorted_L)}}} E[{Y} | {', '.join(sorted_L)}, X={x_0}] * P({', '.join(sorted_L)} | X={x_1})"
            else:
                raise ValueError("ETT : structure non-identifiable, backdoor non bloquée dans G(x_0).")

        return f"{first} - {second}"

    def swig_sequential_plan_evaluation_criterion(self, graph_swig: nx.DiGraph, ordered_treatments: List, ordered_covariates: List[Set], Y_counterfactual: Any) -> bool:
        """Critère séquentiel de g-identifiabilité d'un plan multi-étapes avec isolation du futur."""
        k = len(ordered_treatments)
        if len(ordered_covariates) != k:
            raise ValueError(f"Plan séquentiel : {k} étapes mais {len(ordered_covariates)} ensembles de covariables.")

        random_vars, fixed_nodes = _partition_nodes(graph_swig)
        if Y_counterfactual not in random_vars:
            raise ValueError(f"Plan séq. : cible Y={Y_counterfactual} doit être probabiliste.")

        for m in range(k):
            X_m = ordered_treatments[m]
            L_m = set(ordered_covariates[m])
            if X_m not in random_vars:
                raise ValueError(f"Plan séq. : traitement '{X_m}' non aléatoire.")
            if not L_m.issubset(random_vars):
                raise ValueError(f"Plan séq. : covariables L_{m} non aléatoires.")
            if not any(n[1] == X_m for n in fixed_nodes):
                raise ValueError(f"Plan séq. : le traitement '{X_m}' n'a pas été scindé.")

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
                if str(_get_base_variable(node)) in horizon:
                    to_purge.add(node)
                    continue
                if _is_counterfactual_node(node):
                    for item in node[1]:
                        if str(_get_base_variable(item)) in future_treats:
                            to_purge.add(node)
                            break

            subgraph = graph_swig.subgraph(all_nodes - to_purge)
            if not self.counterfactual_d_separation(subgraph, {X_m}, {Y_counterfactual}, H_m):
                return False

        return True

    def generate_random_dag(self, node_names: Optional[List[str]] = None, method: str = "erdos") -> "Swig":
        """Génère la topologie d'un DAG aléatoire et l'injecte dans la structure."""
        # Quand node_names est fourni, c'est lui qui définit le nombre de nœuds.
        # Utiliser self.config.num_nodes dans ce cas crée une incohérence (ex : après
        # set_level(2), num_nodes vaut 20 mais node_names n'a que 5 éléments).
        n_nodes = len(node_names) if node_names is not None else self.config.num_nodes
        dag_wrapper = get_random_DAG(
            n_nodes=n_nodes,
            edge_prob=self.config.graph_density,
            node_names=node_names,
            latents=(self.config.num_latents > 0),
            seed=self.config.random_seed,
            method=method
        )

        graph = _extract_graph(dag_wrapper)
        self.add_nodes_from(graph.nodes())
        self.add_edges_from(graph.edges())
        # get_random_DAG reçoit un booléen pour latents — il décide lui-même du
        # nombre de nœuds latents. On le plafonne ici à num_latents.
        raw_latents = set(getattr(dag_wrapper, 'latents', set()))
        self.latents = set(sorted(raw_latents)[:self.config.num_latents])

        if not nx.is_directed_acyclic_graph(self):
            raise RuntimeError("Le DAG généré contient un cycle.")

        self._generate_cpds()
        self.add_cpds(*self.custom_cpds.values())
        if not self.check_model():
            raise RuntimeError("Réseau Bayésien factuel structurellement mal formé.")

        return self

    def _generate_cpds(self) -> None:
        """Génère des lois conditionnelles (CPDs) aléatoires sous contrainte de positivité."""
        self.custom_cpds = {}
        self.cardinalities = {}
        k = self.config.cardinality

        if k * self.config.cpd_low >= 1.0:
            raise ValueError(f"Loi de positivité impossible : k·cpd_low = {k * self.config.cpd_low} ≥ 1.")

        base_dag = nx.DiGraph()
        base_dag.add_nodes_from(self.nodes())
        base_dag.add_edges_from(self.edges())

        for node in nx.topological_sort(base_dag):
            self.cardinalities[node] = k
            parents = list(base_dag.predecessors(node))
            num_parents = len(parents)
            num_combos = max(1, k ** num_parents)

            if k == 2:
                if num_parents == 0:
                    p = random.uniform(self.config.cpd_low, self.config.cpd_high)
                    values = [[1 - p], [p]]
                else:
                    v0, v1 = [], []
                    for _ in range(num_combos):
                        p = random.uniform(self.config.cpd_low, self.config.cpd_high)
                        v0.append(1 - p)
                        v1.append(p)
                    values = [v0, v1]
            else:
                values = [[] for _ in range(k)]
                floor = self.config.cpd_low
                scale = 1.0 - k * floor
                for _ in range(num_combos):
                    raw = np.random.dirichlet([1.0] * k)
                    adjusted = raw * scale + floor
                    for i in range(k):
                        values[i].append(float(adjusted[i]))

            evidence = parents if num_parents > 0 else None
            evidence_card = [k] * num_parents if num_parents > 0 else None
            self.custom_cpds[node] = TabularCPD(variable=node, variable_card=k, values=values, evidence=evidence, evidence_card=evidence_card)

    def transform_to_swig(self, interventions: Dict[str, int]) -> Tuple[nx.DiGraph, DiscreteBayesianNetwork, List[str]]:
        """Construit le SWIG instancié G(ã) en orchestrant les transformations graphiques internes."""
        for var, val in interventions.items():
            if var not in self.nodes():
                raise ValueError(f"Intervention impossible : variable '{var}' absente du DAG G.")
            card = self.cardinalities[var]
            if not isinstance(val, int) or not (0 <= val < card):
                raise ValueError(f"La valeur d'intervention pour '{var}' doit être un entier dans [0, {card-1}], reçu {val!r}.")

        base_dag = nx.DiGraph()
        base_dag.add_nodes_from(self.nodes())
        base_dag.add_edges_from(self.edges())

        treatments = set(interventions.keys())

        star_G = self.node_split_transform(base_dag, treatments)
        fixed_meta = [("fixed", t) for t in treatments]
        swit_G = self.counterfactual_labeling(star_G, fixed_meta)
        swig_graph = self.swig_instantiation(swit_G, interventions)
        swig_bn = self._build_swig_bn(swig_graph, interventions)
        counterfactuals = sorted(_node_to_str(n) for n in swig_graph.nodes() if _is_counterfactual_node(n) and len(n[1]) > 0)

        return swig_graph, swig_bn, counterfactuals

    def _build_swig_bn(self, swig_graph: nx.DiGraph, interventions: Dict[str, int]) -> DiscreteBayesianNetwork:
        """Construit le BN pgmpy associé au SWIG instancié en appliquant la propriété de modularité."""
        node_to_pgmpy: Dict[Any, str] = {}
        for n in swig_graph.nodes():
            if isinstance(n, tuple) and n[0] == "concrete_fixed":
                node_to_pgmpy[n] = f"x_{n[1]}"
            else:
                node_to_pgmpy[n] = _node_to_str(n)

        bn = DiscreteBayesianNetwork()
        bn.add_nodes_from(node_to_pgmpy.values())
        for u, v in swig_graph.edges():
            bn.add_edge(node_to_pgmpy[u], node_to_pgmpy[v])

        for n in swig_graph.nodes():
            if isinstance(n, tuple) and n[0] == "concrete_fixed":
                _, t, val = n
                card = self.cardinalities[t]
                values = [[0.0] for _ in range(card)]
                values[val] = [1.0]
                bn.add_cpds(TabularCPD(variable=node_to_pgmpy[n], variable_card=card, values=values))

        # Index base_var → nœud SWIG (non-fixe) pour éviter la boucle O(n²)
        base_to_swig: Dict[Any, Any] = {
            _get_base_variable(nn): nn
            for nn in swig_graph.nodes()
            if not _is_fixed_node(nn)
        }

        for n in swig_graph.nodes():
            if _is_fixed_node(n):
                continue
            base = _get_base_variable(n)
            original_cpd = self.custom_cpds[base]
            card = self.cardinalities[base]

            original_parents = list(original_cpd.variables[1:])
            new_evidence: List[str] = []
            for orig_p in original_parents:
                if orig_p in interventions:
                    fixed_tuple = ("concrete_fixed", orig_p, interventions[orig_p])
                    if fixed_tuple not in node_to_pgmpy:
                        raise RuntimeError(f"Rupture de mapping SWIG BN : nœud fixe absent : {fixed_tuple}")
                    new_evidence.append(node_to_pgmpy[fixed_tuple])
                else:
                    found = base_to_swig.get(orig_p)
                    if found is None:
                        raise RuntimeError(f"Échec de résolution topologique du parent SWIG BN '{orig_p}'.")
                    new_evidence.append(node_to_pgmpy[found])

            if not new_evidence:
                cpd = TabularCPD(variable=node_to_pgmpy[n], variable_card=card, values=original_cpd.get_values())
            else:
                cpd = TabularCPD(variable=node_to_pgmpy[n], variable_card=card, values=original_cpd.get_values(), evidence=new_evidence, evidence_card=list(original_cpd.cardinality[1:]))
            bn.add_cpds(cpd)

        if not bn.check_model():
            raise RuntimeError("Le sous-réseau bayésien SWIG pgmpy généré viole la cohérence probabiliste.")
        return bn

    def query_probability(self, swig_bn: DiscreteBayesianNetwork, target: str, state: int = 1, evidence: Optional[Dict[str, int]] = None) -> float:
        """Exécute l'inférence par élimination de variables sur le réseau d'un monde unique."""
        if target not in swig_bn.nodes():
            raise ValueError(f"La variable cible '{target}' n'est pas résoluble dans ce monde unique.")
        inf = VariableElimination(swig_bn)
        result = inf.query(variables=[target], evidence=evidence, show_progress=False)
        return float(round(result.values[state], 6))

    def check_d_separation(self, swig_graph: nx.DiGraph, target: Any, treatment: Any, observed_vars: Optional[List] = None, forbid_latents: bool = True) -> bool:
        """Évalue l'indépendance contrefactuelle target ⊥⊥ treatment | observed_vars sur l'espace épuré."""
        observed = list(observed_vars or [])
        if forbid_latents:
            for obs in observed:
                base = _get_base_variable(obs)
                if str(base) in self.latents:
                    raise ValueError(f"Rupture d'observabilité : impossible de conditionner l'ancêtre latent '{obs}'.")
        return self.counterfactual_d_separation(swig_graph, {treatment}, {target}, set(observed))

    def render_graph_description(self) -> str:
        """Génère la description sémantique factuelle de la topologie du DAG d'origine."""
        return " ".join(
            f"Node {n} points to {', '.join(map(str, sorted(self.successors(n))))}."
            if self.out_degree(n) > 0
            else f"Node {n} has no outgoing links."
            for n in sorted(self.nodes())
        )

    def cpds_text(self) -> str:
        """Renvoie l'expression textuelle de toutes les matrices de probabilités conditionnelles factuelles."""
        return "\n".join(str(self.custom_cpds[n]) for n in sorted(self.custom_cpds.keys()))


def _score_prob(answer: str, entry) -> float:
    """Scoring partagé : tolérance ±0.05 → 1.0, ±0.10 → 0.5, sinon 0.0."""
    try:
        match = re.search(r"_!_(.*?)_!_", answer)
        pred_str = match.group(1).strip() if match else None
        if pred_str is None:
            # Capture 0, 1, 0.xxx ou 1.xxx en fallback
            m = re.search(r"\b(1(?:\.\d+)?|0(?:\.\d+)?)\b", answer)
            pred_str = m.group() if m else None
        if pred_str is None:
            return 0.0
        d = abs(float(pred_str) - float(entry.answer))
        return 1.0 if d <= 0.05 else (0.5 if d <= 0.10 else 0.0)
    except Exception:
        return 0.0


class CancerEmpiricalTask(Task):
    """Tâche médicale empirique avec vocabulaire clinique."""

    def __init__(self, config: Optional[SwigConfig] = None):
        # deepcopy obligatoire : sans lui, on mute l'objet passé par l'appelant.
        # Exemple : cfg = SwigConfig(num_nodes=8) ; task = CancerEmpiricalTask(cfg)
        # → cfg.num_nodes vaudrait silencieusement 5 après __init__.
        cfg = copy.deepcopy(config) if config is not None else SwigConfig()
        cfg.num_latents = 0
        cfg.num_nodes = 5
        super().__init__(cfg)

    def generate(self):
        model = Swig(self.config)
        n = min(self.config.num_nodes, len(_MEDICAL_NODES))
        medical_nodes = _MEDICAL_NODES[:n]
        model.generate_random_dag(node_names=medical_nodes)

        v_star = self.config.intervention_value
        v_star = v_star if v_star < model.cardinalities["Traitement"] else 0
        interventions = {"Traitement": v_star}

        swig_graph, swig_bn, counterfactuals = model.transform_to_swig(interventions)


        if not counterfactuals:
            return None
        cancer_targets = [c for c in counterfactuals if c.startswith("Cancer")]
        if not cancer_targets:
            return None
        target = cancer_targets[0]

        try:
            df = model.simulate(n_samples=5000, show_progress=False)
        except Exception:
            return None
        freq = df.value_counts().reset_index(name="count").to_string(index=False)
        true_prob = model.query_probability(swig_bn, target, state=1)

        metadata = {
            "graph_description": model.render_graph_description(),
            "interventions": interventions,
            "target": target,
            "dataset_summary": freq,
            "exact_theoretical_prob": true_prob,
            "cot": f"Clinical truth: P({target}=1) = {true_prob}",
        }

        return Problem(metadata=metadata, answer=str(true_prob))

    def prompt(self, m):
        v_star = list(m["interventions"].values())[0]
        return (
            f"Clinical graph:\n{m['graph_description']}\n\n"
            f"5000-patient data:\n```\n{m['dataset_summary']}\n```\n\n"
            f"Force Traitement={v_star}. Estimate P({m['target']}=1) via empirical g-formula.\n"
            f"Wrap answer: _!_value_!_"
        )

    def score_answer(self, answer, entry):
        return _score_prob(answer, entry)


class CounterfactualPatientTask(Task):

    def __init__(self, config: Optional[SwigConfig] = None):
        cfg = copy.deepcopy(config) if config is not None else SwigConfig()
        cfg.num_latents = 0
        cfg.num_nodes = 5
        super().__init__(cfg)

    def generate(self):
        model = Swig(self.config)
        n = min(self.config.num_nodes, len(_MEDICAL_NODES))
        medical_nodes = _MEDICAL_NODES[:n]
        model.generate_random_dag(node_names=medical_nodes)

        try:
            df = model.simulate(n_samples=1, show_progress=False)
        except Exception:
            return None

        patient_row = df.iloc[0]
        obs = {col: int(patient_row[col]) for col in medical_nodes if col in patient_row.index}

        obs_treatment = obs.get("Traitement", 0)
        cf_treatment = 1 - obs_treatment
        interventions = {"Traitement": cf_treatment}

        swig_graph, swig_bn, counterfactuals = model.transform_to_swig(interventions)
        cancer_targets = [c for c in counterfactuals if c.startswith("Cancer")]
        if not cancer_targets:
            return None
        target = cancer_targets[0]

        # Variables pré-traitement : non-descendants de Traitement dans le DAG original
        treatment_descendants = nx.descendants(model, "Traitement") | {"Traitement"}
        pre_treat_vars = [
            n for n in model.nodes()
            if n not in treatment_descendants and str(n) not in model.latents
        ]

        # Evidence : valeurs observées des variables pré-traitement présentes dans le BN SWIG
        swig_bn_nodes = set(swig_bn.nodes())
        evidence = {
            str(v): obs[str(v)]
            for v in pre_treat_vars
            if str(v) in obs and str(v) in swig_bn_nodes
        }

        try:
            cf_prob = model.query_probability(swig_bn, target, state=1, evidence=evidence or None)
        except Exception:
            return None

        pre_str = ", ".join(f"{k}={v}" for k, v in evidence.items()) if evidence else "∅"
        metadata = {
            "graph_description": model.render_graph_description(),
            "patient_obs": obs,
            "obs_treatment": obs_treatment,
            "cf_treatment": cf_treatment,
            "pre_treatment_vars": [str(v) for v in pre_treat_vars],
            "evidence": evidence,
            "target": target,
            "interventions": interventions,
            "exact_theoretical_prob": cf_prob,
            "cot": f"P({target}=1 | {pre_str}) = {cf_prob}",
        }
        draw_dag_and_swig(model, swig_graph, swig_bn=swig_bn)
        return Problem(metadata=metadata, answer=str(cf_prob))

    def prompt(self, m):
        obs = m["patient_obs"]
        obs_str = ", ".join(f"${k}={v}$" for k, v in obs.items())
        cf, obs_t = m["cf_treatment"], m["obs_treatment"]
        target = m["target"]
        ev_str = (
            ", ".join(f"${k}={v}$" for k, v in m["evidence"].items())
            if m["evidence"] else "no pre-treatment covariates"
        )
        return (
            f"Causal graph:\n{m['graph_description']}\n\n"
            f"Observed patient: {obs_str}.\n"
            f"This patient received $\\text{{Traitement}}={obs_t}$ "
            f"and had $\\text{{Cancer}}={obs['Cancer']}$.\n\n"
            f"Counterfactual question: if we had forced $\\text{{Traitement}}={cf}$ in the past, "
            f"what is $P({target}=1 \\mid {ev_str})$ for this specific patient?\n\n"
            f"Wrap your answer: _!_value_!_"
        )

    def score_answer(self, answer, entry):
        return _score_prob(answer, entry)


class ProbabilityOfNecessityTask(Task):

    def __init__(self, config: Optional[SwigConfig] = None):
        cfg = copy.deepcopy(config) if config is not None else SwigConfig()
        cfg.num_latents = 0
        cfg.num_nodes = 5
        super().__init__(cfg)

    def generate(self):
        model = Swig(self.config)
        n = min(self.config.num_nodes, len(_MEDICAL_NODES))
        medical_nodes = _MEDICAL_NODES[:n]
        model.generate_random_dag(node_names=medical_nodes)

        # Construire les deux SWIGs pour T=1 et T=0
        swig_graph_t1, swig_bn_t1, cf_t1 = model.transform_to_swig({"Traitement": 1})
        swig_graph_t0, swig_bn_t0, cf_t0 = model.transform_to_swig({"Traitement": 0})

        cancer_t1 = [c for c in cf_t1 if c.startswith("Cancer")]
        cancer_t0 = [c for c in cf_t0 if c.startswith("Cancer")]
        if not cancer_t1 or not cancer_t0:
            return None

        target_t1 = cancer_t1[0]  # Cancer(Traitement=1)
        target_t0 = cancer_t0[0]  # Cancer(Traitement=0)

        # Variables pré-traitement
        treatment_descendants = nx.descendants(model, "Traitement") | {"Traitement"}
        pre_treat_vars = [
            n for n in model.nodes()
            if n not in treatment_descendants and str(n) not in model.latents
        ]

        # Simuler pour trouver un patient avec Traitement=1 et Cancer=1
        try:
            df = model.simulate(n_samples=500, show_progress=False)
        except Exception:
            return None

        matching = df[(df.get("Traitement", -1) == 1) & (df.get("Cancer", -1) == 1)]
        if matching.empty:
            return None

        patient_row = matching.iloc[0]
        obs = {col: int(patient_row[col]) for col in medical_nodes if col in patient_row.index}

        # Evidence sur les variables pré-traitement (même ensemble pour les deux SWIGs)
        swig_nodes = set(swig_bn_t1.nodes())
        evidence = {
            str(v): obs[str(v)]
            for v in pre_treat_vars
            if str(v) in obs and str(v) in swig_nodes
        }

        try:
            p1 = model.query_probability(swig_bn_t1, target_t1, state=1, evidence=evidence or None)
            p0 = model.query_probability(swig_bn_t0, target_t0, state=1, evidence=evidence or None)
        except Exception:
            return None

        if p1 <= 0:
            return None

        # Formule PN sous l'hypothèse de bruit indépendant (exacte pour les CPDs tabulaires)
        pn = round(max(0.0, p1 - p0) / p1, 6)

        pre_str = ", ".join(f"{k}={v}" for k, v in evidence.items()) if evidence else "∅"
        metadata = {
            "graph_description": model.render_graph_description(),
            "graph_cpds": model.cpds_text(),
            "patient_obs": obs,
            "pre_treatment_vars": [str(v) for v in pre_treat_vars],
            "evidence": evidence,
            "target_t1": target_t1,
            "target_t0": target_t0,
            "p_cancer_do_t1": p1,
            "p_cancer_do_t0": p0,
            "exact_theoretical_prob": pn,
            "cot": (
                f"Step 1: Read the causal graph and CPDs to identify the parents of Cancer "
                f"in each intervened world.\n"
                f"Step 2: Compute P(Cancer=1 | do(Traitement=1), {pre_str}) = {p1} "
                f"by marginalizing over the SWIG(do=1) distribution.\n"
                f"Step 3: Compute P(Cancer=1 | do(Traitement=0), {pre_str}) = {p0} "
                f"by marginalizing over the SWIG(do=0) distribution.\n"
                f"Step 4: PN = max(0, {p1} - {p0}) / {p1} = {pn}"
            ),
        }
        draw_dag_and_two_swigs(model, swig_graph_t1, swig_graph_t0,
                               swig_bn_t1=swig_bn_t1, swig_bn_t0=swig_bn_t0)
        return Problem(metadata=metadata, answer=str(pn))

    def prompt(self, m):
        obs = m["patient_obs"]
        obs_str = ", ".join(f"{k}={v}" for k, v in obs.items())
        ev_str = (
            ", ".join(f"{k}={v}" for k, v in m["evidence"].items())
            if m["evidence"] else "no pre-treatment covariates"
        )
        return (
            f"Causal graph structure:\n{m['graph_description']}\n\n"
            f"Conditional probability tables:\n{m['graph_cpds']}\n\n"
            f"Observed patient: {obs_str}.\n"
            f"This patient received Traitement=1 and developed Cancer=1.\n\n"
            f"Task: Compute the Probability of Necessity (PN) — the probability that "
            f"this patient's cancer would NOT have occurred had they NOT received treatment.\n\n"
            f"Formally: PN = P(Cancer(Traitement=0)=0 | Traitement=1, Cancer=1, {ev_str})\n\n"
            f"Under the independent background noise assumption:\n"
            f"  PN = max(0, p1 - p0) / p1\n"
            f"where p1 = P(Cancer=1 | do(Traitement=1), {ev_str})\n"
            f"  and p0 = P(Cancer=1 | do(Traitement=0), {ev_str})\n\n"
            f"Compute p1 and p0 from the causal graph and CPDs above "
            f"(intervening on Traitement removes all incoming edges to it), "
            f"then apply the formula.\n"
            f"Wrap your final answer: _!_value_!_"
        )

    def score_answer(self, answer, entry):
        return _score_prob(answer, entry)
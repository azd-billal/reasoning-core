from ast import literal_eval

import networkx as nx
import random

from networkx.generators import directed

from reasoning_core.template import Task, Problem, Config
from dataclasses import dataclass

@dataclass
class SwigConfig(Config):
    num_nodes: int = 5
    graph_density: float = 0.4 # proba d'avoir un lien entre deux noeuds
    leaf_intervention_prob: float = 0.2 # noeuds sans enfant (donc piège)

    def update(self, c):
        self.num_nodes *= (1 + c)

def _parse_list(x):
    try:
        x = literal_eval(x)
        return x if isinstance(x, list) else None
    except Exception:
        return None


class Swig:

    def __init__(self, config=SwigConfig()):
        self.config = config
        self.graph = nx.DiGraph()


    def _make_dag(self):

        n = self.config.num_nodes
        p = self.config.graph_density


        G = nx.fast_gnp_random_graph(n, p, directed=True)


        nodes = list(G.nodes())
        random.shuffle(nodes)
        order = {n: i for i, n in enumerate(nodes)}
        edges_to_remove = [(u, v) for u, v in G.edges() if order[u] >= order[v]]
        G.remove_edges_from(edges_to_remove)

        if G.number_of_edges() == 0 and n > 1:
            return self._make_dag()

        self.graph = nx.convert_node_labels_to_integers(G)
        return self.graph

    def _render_graph(self):
        return " ".join(
            f"Node {n} points to {', '.join(map(str, sorted(self.graph.successors(n))))}."
            if self.graph.out_degree(n) > 0 else f"Node {n} has no outgoing links."
            for n in sorted(self.graph.nodes())
        )

class SwigInterventionTask(Task):

    def __init__(self, config=SwigConfig()):
        super().__init__(config)

    def make_cot(self, G, target, intervention, descendants):
        lines = [
            f"Goal: Identify counterfactual variables after intervention {intervention} on Node {target}.",
            f"1. Target of intervention: {target}",
            f"2. Intervention type: Fixed value forced by {intervention}."
        ]

        if not descendants:
            lines.append(f"3. Node {target} is a leaf node (no outgoing edges).")
            lines.append("4. Result: No other variables are affected. Final list: [].")
        else:
            lines.append(f"3. Find all nodes reachable from Node {target} (descendants).")
            lines.append(f"4. Descendants found: {sorted(list(descendants))}")
            lines.append(f"5. Apply SWIG naming rule: Each descendant Y becomes Y({intervention}).")
            lines.append(f"6. Final counterfactual set: {[f'{d}({intervention})' for d in sorted(descendants)]}")

        return "\n".join(lines)

    def generate(self):

        swig_model = Swig(self.config)
        G = swig_model._make_dag()

        normals_nodes = [n for n in G.nodes() if G.out_degree(n) > 0]
        leafs_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

        if (random.random() < self.config.leaf_intervention_prob) and leafs_nodes:
            query = random.choice(leafs_nodes)
        else:
            query = random.choice(normals_nodes) if normals_nodes else random.choice(list(G.nodes()))

        descendants = sorted(list(nx.descendants(G, query)))
        intervention = f"do({query})"
        answer = [f"{d}({intervention})" for d in descendants]

        metadata = {
            "graph_description": swig_model._render_graph(),
            "query" : query,
            "intervention": intervention,
            "nodes": list(G.nodes()),
            "edges": list(G.edges()),
            "cot": self.make_cot(G, query, intervention, descendants)
        }

        nx.draw(G)
        return Problem(metadata=metadata, answer=str(answer))



    def prompt(self, m):
        return (
            f"Consider the directed causal graph:\n\n{m['graph_description']}\n\n"
            f"We perform a surgical intervention on Node {m['query']} by forcing its value: {m['intervention']}.\n"
            f"List all new counterfactual variables created by this intervention.\n"
            "The answer is a Python list of strings (e.g., ['1(do(0))']), or [] if no variables are affected."
        )

    def score_answer(self, answer, entry):
        pred = _parse_list(answer)
        true = _parse_list(entry.answer)

        if pred is None or true is None:
            return 0.0

        set_pred = set(pred)
        set_true = set(true)

        if not set_pred and not set_true:
            return 1.0

        if not set_pred or not set_true:
            return 0.0

        intersection = set_pred.intersection(true)
        union = set_pred.union(true)

        score = len(intersection) / len(union)

        return round(score,2)
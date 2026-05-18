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
    leaf_intervention_prob: float = 0.2 # noeuds sans enfant

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

    def transform_to_swig(self, interventions):
        """
        Applique l'algorithme strict du SWIG pour des interventions multiples.
        interventions: dictionnaire {str(node_id): intervention_val} (ex: {'0': 'x0', '1': 'x1'})
        """
        swig_G = nx.DiGraph()
        fixed_nodes = set(interventions.values())

        # 1. Node Splitting & Redirection des arêtes
        for u, v in self.graph.edges():
            str_u = str(u)
            str_v = str(v)
            # Si la source est intervenue, l'arête part de sa moitié fixe, sinon de sa moitié aléatoire
            new_u = interventions[str_u] if str_u in interventions else str_u
            # La destination reçoit toujours l'arête sur sa moitié aléatoire
            new_v = str_v
            swig_G.add_edge(new_u, new_v)

        # Garantir la présence de tous les nœuds (y compris isolés ou moitiés de split)
        for n in self.graph.nodes():
            str_n = str(n)
            if str_n not in swig_G:
                swig_G.add_node(str_n)
            if str_n in interventions:
                fixed_val = interventions[str_n]
                if fixed_val not in swig_G:
                    swig_G.add_node(fixed_val)

        # 2. Labeling : Pour chaque nœud aléatoire, trouver ses ancêtres fixes
        random_nodes = [n for n in swig_G.nodes() if n not in fixed_nodes]

        counterfactuals = []
        for v in random_nodes:
            ancestors = nx.ancestors(swig_G, v)
            fixed_ancestors = ancestors.intersection(fixed_nodes)
            if fixed_ancestors:
                # Si plusieurs interventions sont ancêtres, on les sépare par une virgule, triées
                a_v = ",".join(sorted(list(fixed_ancestors)))
                counterfactuals.append(f"{v}({a_v})")

        return swig_G, sorted(counterfactuals)

class SwigInterventionTask(Task):

    def __init__(self, config=SwigConfig()):
        super().__init__(config)

    def make_cot(self, interventions, counterfactuals):
        lines = [
            "Goal: Identify counterfactual variables after a joint intervention on multiple nodes.",
            "1. Joint Interventions: " + ", ".join([f"Node {n} set to '{val}'" for n, val in interventions.items()]),
            "2. Multi-Node Splitting: Each targeted node is split. Incoming edges go to the random component, outgoing edges leave from the fixed component.",
            "3. Ancestor Detection: Trace backward paths in the new graph to find which fixed values reach each random node."
        ]

        if not counterfactuals:
            lines.append("4. Result: No variables are descendants of the fixed intervention nodes. Final list: [].")
        else:
            lines.append("4. Labeling: Applied the SWIG naming rule V(a_V) based on discovered fixed ancestors.")
            lines.append(f"5. Final counterfactual set: {counterfactuals}")

        return "\n".join(lines)

    def generate(self):
        swig_model = Swig(self.config)
        G = swig_model._make_dag()

        # Choix dynamique d'un nombre d'interventions simultanées (entre 1 et 2 pour ne pas surcharger)
        num_interventions = random.randint(1, min(2, len(G.nodes())))
        chosen_nodes = random.sample(list(G.nodes()), num_interventions)

        # On force les clés en string pour éviter le crash d'EasyDict
        interventions = {str(n): f"x{n}" for n in sorted(chosen_nodes)}

        swig_G, answer = swig_model.transform_to_swig(interventions)

        metadata = {
            "graph_description": swig_model._render_graph(),
            "interventions": interventions,
            "nodes": list(G.nodes()),
            "edges": list(G.edges()),
            "cot": self.make_cot(interventions, answer)
        }

        nx.draw(swig_G, with_labels=True, node_color='lightblue', font_weight='bold', node_size=800)
        return Problem(metadata=metadata, answer=str(answer))

    def prompt(self, m):
        interv_desc = ", ".join([f"Node {n} to '{val}'" for n, val in m['interventions'].items()])
        return (
            f"Consider the directed causal graph:\n\n{m['graph_description']}\n\n"
            f"We perform a simultaneous joint surgical intervention setting: {interv_desc}.\n"
            f"Apply the Single-World Intervention Graph (SWIG) node-splitting transformation.\n"
            f"List all new counterfactual variables created by this joint intervention (format: Node(fixed_ancestors)).\n"
            "The answer must be a Python list of strings (e.g., ['1(x0)', '2(x0,x1)']), or [] if no variables are affected."
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

        intersection = set_pred.intersection(set_true)
        union = set_pred.union(set_true)

        score = len(intersection) / len(union)

        return round(score, 2)
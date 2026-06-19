"""Tache de raisonnement contrefactuel de couche L2.25 (CBN2.25).

Reference :  « A Hierarchy of Graphical Models » (Def. 11 / Ex. 5),
qui situe la couche L2.25 entre l'interventionnel (rung 2) et le contrefactuel
plein (rung 3) : une intervention `{X = x}` propage la MEME valeur `x` a tous les
descendants de X (condition ii), ce qui correspond exactement au node-splitting
d'un SWIG.

La tache : etant donne le SCM (DAG causal + tables de probabilite, via le SWIG),
le SWIG sous `do(X = x)`, et un jeu d'observations factuelles, calculer la
probabilite contrefactuelle `P(Y(X=x) = 1 | observations)`. La verite-terrain est
calculee exactement par `SwigCounterfactualEngine.compute_cbn_225` (Balke & Pearl
1994, abduction-action-prediction).
"""

from dataclasses import dataclass
from typing import Optional

from reasoning_core.template import Problem, Task, edict, Config
from reasoning_core.utils import score_scalar
from reasoning_core.tasks.swig import Swig, SwigCounterfactualEngine


@dataclass
class ConfigR225(Config):
    num_nodes: int = 5
    graph_density: float = 0.4
    cardinality: int = 2
    max_response_functions: int = 200_000
    random_seed: Optional[int] = None

    def update(self, c):
        # Difficulte : graphe plus grand et plus dense (plus de chemins
        # contrefactuels), donc abduction-action-prediction plus exigeante.
        self.num_nodes += c
        self.graph_density = min(0.8, self.graph_density + 0.05 * c)


class R225(Task):
    """Raisonnement contrefactuel L2.25 sur un SWIG : calculer
    `P(Y(X=x) = 1 | observations)` a partir du SCM, du SWIG et des observations.

    Le SCM source est markovien (sans latente, requis par le moteur exact). La
    reponse est une probabilite scalaire, notee par `score_scalar`.
    """

    def __init__(self, config=ConfigR225()):
        super().__init__(config=config)

    def generate(self):
        swig = Swig(self.config)
        try:
            query = swig.generate_225()
            engine = SwigCounterfactualEngine(swig)
            target, observations = query["target"], query["observations"]
            distribution = engine.compute_cbn_225(target, observations)
            cot = engine.get_cot(target, observations, target_state=1)
        except (RuntimeError, ValueError):
            return None

        answer = round(float(distribution.get(1, 0.0)), 6)
        meta = edict(
            swig_description=query["swig_description"],
            situation_initiale=query["situation_initiale"],
            observations=observations,
            target=target,
            cot=cot,
        )
        return Problem(metadata=meta, answer=f"{answer:.6f}")

    def prompt(self, metadata):
        do_txt = ", ".join(f"{k}={v}" for k, v in sorted(metadata.situation_initiale.items()))
        obs_txt = (
            ", ".join(f"{k}={v}" for k, v in sorted(metadata.observations.items()))
            or "none"
        )
        return (
            f"{metadata.swig_description}\n\n"
            f"In this Single-World Intervention Graph for the intervention do({do_txt}), a node "
            f"written like {metadata.target} is the counterfactual value of its source variable "
            f"in the world where do({do_txt}) is enforced; un-indexed nodes keep their natural "
            f"(factual) value, and the conditional probability tables above give the mechanism of "
            f"every source variable.\n\n"
            f"Factual observations in the real (un-intervened) world: {obs_txt}.\n\n"
            f"Question: in the counterfactual world under do({do_txt}), and given the factual "
            f"observations above, what is the probability that {metadata.target} = 1?\n"
            f"Reason by abduction (update beliefs on the exogenous noise from the observations), "
            f"action (apply do({do_txt})), then prediction.\n\n"
            f"Answer with only the probability value (a number in [0, 1])."
        )

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)

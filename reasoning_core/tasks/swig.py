from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from reasoning_core.template import Config
from reasoning_core.tasks._causal_utils import to_nl_DBN, to_nl_CPD


class SwigConfig(Config):
    num_nodes: int = 5
    graph_density: float = 0.4
    num_latents: int = 1
    cardinality: int = 2
    random_seed: Optional[int] = None
    max_response_functions: int = 200_000

    def update(self, c: float) -> None:
        self.num_nodes = max(2, int(self.num_nodes * (1 + c)))
        self.graph_density = min(0.8, self.graph_density + 0.1 * c)
        if self.num_latents > 0:
            self.num_latents = max(1, int(self.num_latents * (1 + c)))


class Swig(DiscreteBayesianNetwork):
    """Single-World Intervention Graph (Richardson & Robins, 2013).

    Le reseau bayesien causal source est tire DIRECTEMENT (DAG + CPDs) par pgmpy
    `DiscreteBayesianNetwork.get_random`, puis transforme par node-splitting :
    chaque variable d'intervention X est scindee en un noeud FIXE `do(X=v)` et un
    noeud ALEATOIRE indexe `Y(x)`.

    SWIG SIMPLIFIE : l'objet ne stocke QUE le SCM source `self.source_bn` (avec ses
    CPDs) et l'intervention `self.intervention` (dict `{X: x}`, "la partie fixee").
    On ne fait PLUS de node-splitting : aucun graphe scinde, aucun label `Y(X=v)`,
    aucune CPD materialisee. Le monde contrefactuel est entierement determine par
    (source_bn, intervention) ; quelles variables heritent la valeur x (les
    descendants de X) se lit a la demande sur `source_bn`.

    Repartition des roles (volontaire) :
      - `self.source_bn` + `self.intervention` SONT le SWIG ;
      - le **SCM source** (`self.source_bn`, avec ses CPDs) porte les mecanismes ;
      - le **calcul numerique** des contrefactuels n'est PAS dans cette classe : il
        est fait par `SwigCounterfactualEngine`, qui lit `source_bn` + `intervention`.

    Surface : `generate_random_swig` (tirage du SCM + choix de l'intervention),
    `_random_intervention` (intervention au hasard) et `generate_225` (construit une
    requete L2.25 et renvoie sa formule LaTeX).
    """

    def __init__(
            self,
            config: Optional[SwigConfig] = None,
            *,
            bn: Optional[DiscreteBayesianNetwork] = None,
    ):
        super().__init__()
        self.config = config if config is not None else SwigConfig()
        self._rng = random.Random(self.config.random_seed)
        # Un SWIG n'est QUE (source_bn, intervention) : on stocke ici l'intervention
        # do(X=x), "la partie fixee". Aucun graphe scinde, aucune CPD materialisee.
        self.intervention: Dict[str, Any] = {}
        # BN source (SCM). Peut etre fourni au constructeur (`bn=`) pour imposer le
        # reseau voulu ; sinon rempli par generate_random_swig (tirage aleatoire).
        self.source_bn: Optional[DiscreteBayesianNetwork] = bn
        # Requete L2.25 stockee par generate_225 (la formule LaTeX est renvoyee).
        self.target: Optional[str] = None
        self.observations: Optional[Dict[str, Any]] = None
        self.situation_initiale: Optional[Dict[str, Any]] = None
        self.swig_description: Optional[str] = None
        self.engine: Optional["SwigCounterfactualEngine"] = None

    def generate_random_swig(
            self,
            nb_nodes: Optional[int] = None,
            prob_link: Optional[float] = None,
            *,
            interventions: Optional[Mapping[str, Any]] = None,
            node_names: Optional[List[str]] = None,
            cardinality: Optional[int] = None,
            bn: Optional[DiscreteBayesianNetwork] = None,
    ) -> "Swig":
        """Construit un SWIG aleatoire de bout en bout.

        1. BN source : `bn` (argument) s'il est fourni, sinon `self.source_bn`
           (BN impose au constructeur via `bn=`), sinon tire par pgmpy
           `DiscreteBayesianNetwork.get_random` (DAG + CPDs, cardinalite
           `config.cardinality`).
        2. Intervention : `interventions` si fournie ; SINON choisie AU HASARD
           (un noeud ayant des descendants + une valeur au hasard). Le graphe etant
           tire aleatoirement, on ne connait pas sa structure a l'avance : on ne peut
           donc pas exiger l'intervention du dehors, on la tire ici.
        3. On STOCKE simplement (`self.source_bn`, `self.intervention`) -- pas de
           node-splitting, "la partie fixee" est juste l'intervention.

        Un SWIG est toujours defini par une intervention (jamais de "mode source").

        Parametres (defauts dans `self.config`) : `nb_nodes`, `prob_link`
        (densite d'aretes), `node_names` (sa longueur prime sur nb_nodes),
        `cardinality`, `bn` (BN source impose, p.ex. pour tests).
        """
        if nb_nodes is not None:
            self.config.num_nodes = nb_nodes
        if prob_link is not None:
            self.config.graph_density = prob_link
        if cardinality is not None:
            self.config.cardinality = cardinality

        if bn is None:
            bn = self.source_bn   # BN impose au constructeur, si fourni
        if bn is None:
            k = self.config.cardinality
            n_nodes = len(node_names) if node_names is not None else self.config.num_nodes
            bn = DiscreteBayesianNetwork.get_random(
                n_nodes=n_nodes,
                edge_prob=self.config.graph_density,
                node_names=[str(x) for x in node_names] if node_names is not None else None,
                n_states=k,
                seed=self._rng.randrange(2**31),
            )

        if interventions is None:
            interventions = self._random_intervention(bn)
        # Node-splitting reduit a "mettre l'intervention dans une variable" : on stocke
        # juste le SCM source et l'intervention do(X=x), rien d'autre.
        self.source_bn = bn
        normalized = {str(var): value for var, value in dict(interventions).items()}
        missing = set(normalized) - {str(n) for n in bn.nodes()}
        if missing:
            raise ValueError(f"Unknown intervention variables: {sorted(missing)}")
        self.intervention = normalized
        return self

    def _random_intervention(self, bn: DiscreteBayesianNetwork) -> Dict[str, Any]:
        """Choisit au hasard une intervention do(X=v) valide : X doit avoir au moins
        un descendant (sinon la requete contrefactuelle serait triviale). Lu
        DIRECTEMENT sur `bn` (aucun graphe reconstruit)."""
        candidates = sorted(n for n in bn.nodes() if bn.out_degree(n) > 0)
        if not candidates:
            raise ValueError("Aucune variable avec descendant : SWIG trivial.")
        x_var = self._rng.choice(candidates)
        x_states = list(bn.get_cpds(x_var).state_names[x_var])
        return {str(x_var): self._rng.choice(x_states)}

    def generate_225(
            self,
            *,
            intervention: Optional[Mapping[str, Any]] = None,
            target: Optional[Any] = None,
            observations: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Construit une requete L2.25 sur ce SWIG, la STOCKE sur l'objet
        (`self.target`, `self.observations`, `self.situation_initiale`,
        `self.swig_description`, `self.engine`) et RENVOIE sa formule en LaTeX
        (ex. `P\\left(Y_{X=x} \\mid Z=z\\right)`). La REPONSE numerique se calcule
        a part avec `self.engine.compute_cbn_225()`.

        Une intervention `{X = x}` definit le monde contrefactuel : par la condition
        (ii) de la Def. 11, la MEME valeur `x` se propage a TOUS les descendants de X.

        L2.25 STRICT : la requete ne conditionne QUE sur des variables NON descendantes
        de l'intervention (Def. 11, Ex. 6). Conditionner sur la valeur factuelle d'un
        descendant de X donnerait une jointe inter-mondes (p.ex. P(Y_x, M)) = L3, hors
        L2.25. La valeur factuelle de X lui-meme reste permise.

        Parametres (tous optionnels) :
          - `intervention` : si fournie, (re)definit l'intervention sur `self.source_bn` ;
            sinon on reutilise le SWIG deja construit (p.ex. par `generate_random_swig`).
          - `target`       : variable source contrefactuelle interrogee ; tiree au hasard
            parmi les descendants de l'intervention si omise.
          - `observations` : valeurs factuelles observees ; echantillonnees du SCM
            source (hors cible ET hors descendants de l'intervention) si omises. Si
            fournies, lever ValueError si elles incluent un descendant de l'intervention.
        """
        if intervention is not None:
            if self.source_bn is None:
                raise ValueError(
                    "generate_225 avec `intervention` exige un BN source : passer "
                    "`bn=` au constructeur ou appeler generate_random_swig d'abord."
                )
            normalized = {str(k): v for k, v in dict(intervention).items()}
            missing = set(normalized) - {str(n) for n in self.source_bn.nodes()}
            if missing:
                raise ValueError(f"Unknown intervention variables: {sorted(missing)}")
            self.intervention = normalized
        if not self.intervention:
            raise ValueError("generate_225 exige un SWIG construit avec intervention(s).")

        bn = self.source_bn
        source_nodes = [str(n) for n in bn.nodes()]
        node_by_str = {str(n): n for n in bn.nodes()}
        # Descendants de l'intervention, lus DIRECTEMENT sur source_bn : ce sont les
        # variables contrefactuelles (elles heritent la valeur d'intervention x).
        post_treatment: Set[str] = set()
        for x_var in self.intervention:
            post_treatment |= {str(d) for d in nx.descendants(bn, node_by_str[x_var])}

        if target is None:
            candidates = sorted(post_treatment)
            if not candidates:
                raise ValueError("Aucun descendant contrefactuel a interroger.")
            target_source = self._rng.choice(candidates)
        else:
            target_source = str(target)

        # L2.25 STRICT (Def. 11 + Ex. 6 du papier) : on ne conditionne QUE sur des
        # variables NON descendantes de l'intervention. Conditionner sur la valeur
        # FACTUELLE d'un descendant de X donnerait du L3 (hors L2.25). La valeur
        # factuelle de X lui-meme reste permise.
        if observations is None:
            sample = bn.simulate(n_samples=1, show_progress=False).iloc[0]
            obs: Dict[str, Any] = {}
            for var in source_nodes:
                if var == target_source or var in post_treatment or var not in sample.index:
                    continue
                value = sample[var]
                obs[var] = value.item() if hasattr(value, "item") else value
        else:
            obs = {}
            for key, value in observations.items():
                k = str(key)
                if k in post_treatment:
                    raise ValueError(
                        f"Observation L2.25 invalide : {key!r} est un descendant de "
                        f"l'intervention. Conditionner sur sa valeur factuelle donne une "
                        f"requete inter-mondes (L3), hors L2.25."
                    )
                obs[k] = value

        # On STOCKE la requete sur le SWIG...
        self.target = target_source
        self.observations = obs
        self.situation_initiale = dict(self.intervention)
        self.swig_description = to_nl_DBN(self.source_bn)
        self.engine = SwigCounterfactualEngine(self)

        # ... et on RENVOIE sa formule en LaTeX. La valeur d'intervention se propage a
        # tous les descendants, d'ou la notation potential-outcome Y_{X=x}. Nom cible
        # groupe entre accolades : les noms (ex. `X_3`) contiennent deja un `_`, donc
        # `X_3_{...}` serait un double-indice invalide en LaTeX.
        do_sub = ", ".join(f"{var}={val}" for var, val in sorted(self.intervention.items()))
        cond = ", ".join(f"{var}={val}" for var, val in sorted(obs.items()))
        target_latex = f"{{{target_source}}}_{{{do_sub}}}"
        return (
            f"P\\left({target_latex} \\mid {cond}\\right)" if cond
            else f"P\\left({target_latex}\\right)"
        )


@dataclass
class _Mechanism:
    """Mecanisme P(V | parents) du SCM source, vu comme fonction de reponse :
    etats de V, parents, configurations de parents (ordre fixe, pour les fonctions
    de reponse completes) et CPD source."""
    cpd: Any
    parents: List[str]
    states: List[Any]
    configs: List[Dict[str, Any]]
    config_index: Dict[Tuple[Any, ...], int]


class SwigCounterfactualEngine:
    """Moteur d'inference contrefactuelle exact, d'apres Balke & Pearl (1994),
    *Probabilistic evaluation of counterfactual queries*

    Principe (representation par FONCTIONS DE REPONSE, section "Probabilistic vs.
    functional specification" du papier) : chaque mecanisme P(V | Pa) du SCM
    source -- porte par `swig.source_bn`, tire par get_random -- est vu comme une
    variable exogene `r_V` qui choisit, pour CHAQUE configuration des parents, la
    valeur de V. Le monde FACTUEL (reel, sans intervention) et le monde
    CONTREFACTUEL (le SWIG, sous do(X=x)) PARTAGENT ces `r_V`.

    NB sur la methode : on ne CONSTRUIT PAS de reseau jumeau (twin network : deux
    copies du DAG partageant les exogenes, sur lequel on lancerait une inference).
    On utilise l'autre methode equivalente du meme papier -- la representation par
    fonctions de reponse -- evaluee par ENUMERATION EXACTE de l'espace des `r_V` :
    pour chaque tirage de `r_V`, on simule le monde factuel ET le monde do() sous
    LES MEMES `r_V`. Ce partage des `r_V` entre les deux mondes est l'equivalent
    fonctionnel du reseau jumeau (Figure 2), mais aucun graphe jumeau n'est materialise.

    Les trois phases du papier (algorithme p.234) :
      1. ABDUCTION  : conditionner les `r_V` sur l'evidence factuelle observee
                      (ici : ne garder que les tirages dont le monde factuel colle
                      a l'evidence ; leur masse cumulee = P(evidence)) ;
      2. ACTION     : forcer do(X=x) dans le monde contrefactuel (deja encode par
                      le node-splitting du SWIG) ;
      3. PREDICTION : propager les memes `r_V` dans le monde do() et lire la cible.
    Abduction et prediction se calculent en une passe (les deux mondes evalues sous
    le meme tirage de `r_V`). Prior sur les `r_V` : CANONIQUE, derive des CPDs du SCM
    source (reponses independantes entre configurations de parents -- le modele
    markovien standard que produit get_random).
    """

    def __init__(self, swig: Swig, *, max_response_functions: Optional[int] = None):
        if swig.source_bn is None:
            raise ValueError("Le moteur exige swig.source_bn (le SCM source).")
        if not swig.intervention:
            raise ValueError("Le moteur exige un SWIG construit avec intervention(s).")
        if getattr(swig.source_bn, "latents", set()):
            raise ValueError("SCM markovien requis : pas de variable latente.")
        self.swig = swig
        self.source_bn = swig.source_bn
        # Le SWIG n'est que (source_bn, intervention) : on lit tout directement.
        self.intervention: Dict[str, Any] = dict(swig.intervention)
        self.source_nodes: List[str] = [str(n) for n in self.source_bn.nodes()]
        # Plafond d'enumeration : argument explicite sinon `config.max_response_functions`.
        self.max_response_functions = (
            max_response_functions
            if max_response_functions is not None
            else getattr(swig.config, "max_response_functions", 200_000)
        )
        # Journal des operations : on garde la trace des etapes (construction du
        # modele, chaque requete avec sa masse d'abduction et sa distribution).
        self.trace: List[str] = []
        # PHASE 1 (representation) : table des fonctions de reponse par noeud.
        self.mechanisms = self._build_response_functions()
        self._order = self._topological_order()
        # Cache de l'enumeration complete (cf. _iter_response_tables) : evite de
        # regenerer l'espace des fonctions de reponse quand r225 appelle
        # compute_cbn_225 PUIS get_cot sur le meme moteur.
        self._response_tables_cache: Optional[List[Tuple[Dict[str, Any], float]]] = None
        do_txt = ", ".join(f"{v}={x}" for v, x in sorted(self.intervention.items()))
        self.trace.append(
            f"Phase 1 (representation) : modele a fonctions de reponse construit sur "
            f"{len(self.mechanisms)} mecanismes ; intervention do({do_txt})."
        )

    def _build_response_functions(self) -> Dict[str, _Mechanism]:
        mech: Dict[str, _Mechanism] = {}
        for var in self.source_nodes:
            cpd = self.source_bn.get_cpds(var)
            parents = [str(p) for p in cpd.variables[1:]]
            parent_states = [list(cpd.state_names[p]) for p in cpd.variables[1:]]
            # Configurations de parents, ordre fixe = ordre des valeurs d'une
            # fonction de reponse complete (cf. _iter_response_tables).
            configs = [
                dict(zip(parents, combo)) for combo in itertools.product(*parent_states)
            ]
            config_index = {tuple(c[p] for p in parents): i for i, c in enumerate(configs)}
            mech[var] = _Mechanism(
                cpd=cpd,
                parents=parents,
                states=list(cpd.state_names[cpd.variable]),
                configs=configs,
                config_index=config_index,
            )
        return mech

    def _topological_order(self) -> List[str]:
        # Tri topologique lu DIRECTEMENT sur le SCM source (aucun graphe reconstruit).
        return [str(n) for n in nx.topological_sort(self.source_bn)]

    def _prob(self, var: str, value: Any, parent_config: Mapping[str, Any]) -> float:
        """P(var = value | parents = parent_config), lue dans la CPD source."""
        assignment = {var: value, **parent_config}
        return float(self.mechanisms[var].cpd.get_value(**assignment))

    def _response_space_size(self) -> int:
        """Nombre de jeux de tables de reponse completes enumeres : produit, par
        noeud, du nombre de fonctions de reponse `k^(#configurations de parents)`."""
        size = 1
        for var in self.source_nodes:
            mech = self.mechanisms[var]
            size *= len(mech.states) ** len(mech.configs)
        return size

    def _node_response_functions(self, var: str):
        """Genere (table_de_reponse, prior) pour un noeud : prior CANONIQUE derive
        de la CPD, P(r_V) = produit_config P(V=r_V(config)|config)."""
        mech = self.mechanisms[var]
        for response_table in itertools.product(mech.states, repeat=len(mech.configs)):
            prior = 1.0
            for cfg, value in zip(mech.configs, response_table):
                prior *= self._prob(var, value, cfg)
                if prior == 0.0:
                    break
            if prior > 0.0:
                yield response_table, prior

    def _iter_response_tables(self):
        """Renvoie la liste (response_tables, poids) : response_tables = {noeud:
        table_de_reponse} sur le produit des fonctions de reponse de chaque noeud,
        poids = produit des priors.

        MEMOISEE a la 1re construction : `compute_cbn_225` et `_distribution_and_mass`
        (donc query/answer/get_cot) consomment cette meme enumeration ; r225 appelant
        compute_cbn_225 PUIS get_cot, on evite ainsi de regenerer tout l'espace des
        fonctions de reponse au 2e appel (compromis : on materialise la liste, mais
        elle est bornee par `max_response_functions`). Garde-fou commun (protege aussi
        compute_cbn_225) : la 1re construction refuse une enumeration plus grande que
        `max_response_functions`."""
        if self._response_tables_cache is None:
            space = self._response_space_size()
            if space > self.max_response_functions:
                raise RuntimeError(
                    "Espace des fonctions de reponse trop grand "
                    f"({space} > {self.max_response_functions})."
                )
            order = self._order
            rf_lists = {var: list(self._node_response_functions(var)) for var in order}

            def rec(i, tables, weight):
                if i == len(order):
                    yield tables, weight
                    return
                var = order[i]
                for response_table, prior in rf_lists[var]:
                    yield from rec(i + 1, {**tables, var: response_table}, weight * prior)

            self._response_tables_cache = list(rec(0, {}, 1.0))
        return self._response_tables_cache

    def _evaluate_world(self, response_tables, interventions):
        """Evalue le SCM en ordre topologique sous `response_tables` : un noeud
        intervenu prend sa valeur do(), sinon la valeur de sa table de reponse a la
        configuration courante de ses parents."""
        world: Dict[str, Any] = {}
        for var in self._order:
            if var in interventions:
                world[var] = interventions[var]
            else:
                mech = self.mechanisms[var]
                config = tuple(world[p] for p in mech.parents)
                world[var] = response_tables[var][mech.config_index[config]]
        return world

    @staticmethod
    def _world_matches(world: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
        """Le monde est-il compatible avec l'evidence observee ?"""
        return all(world.get(var) == value for var, value in evidence.items())

    def _distribution_and_mass(self, target_source: str, evidence: Mapping[str, Any]) -> Tuple[Dict[Any, float], float]:
        """Coeur des phases 2-3 : renvoie (distribution contrefactuelle normalisee
        de la cible, masse d'evidence P(evidence)). Consomme par `compute_cbn_225`
        (qui complete/arrondit la distribution) et par `get_cot` appele seul (qui a
        besoin de la masse `P(evidence)`). Machinerie de tables de reponse completes
        (garde-fou dans `_iter_response_tables`)."""
        interventions = dict(self.intervention)
        distribution: Dict[Any, float] = {}
        evidence_mass = 0.0
        for response_tables, weight in self._iter_response_tables():
            # PHASE 2 (abduction) : ne retenir que les mondes factuels compatibles
            # avec l'evidence observee ; leur masse cumulee = P(evidence).
            factual = self._evaluate_world(response_tables, interventions={})
            if not self._world_matches(factual, evidence):
                continue
            evidence_mass += weight
            # PHASE 3 (prediction) : valeur de la cible dans le monde do(), sous
            # les memes fonctions de reponse.
            counterfactual = self._evaluate_world(response_tables, interventions=interventions)
            y = counterfactual[target_source]
            distribution[y] = distribution.get(y, 0.0) + weight
        if evidence_mass <= 0.0:
            raise ValueError("Evidence factuelle de probabilite nulle (abduction impossible).")
        normalized = {y: w / evidence_mass for y, w in distribution.items()}
        ev_txt = ", ".join(f"{v}={x}" for v, x in sorted(evidence.items())) or "aucune"
        self.trace.append(
            f"Phases 2-3 ({target_source}* | {ev_txt}) : P(evidence)={evidence_mass:.6f}, "
            f"distribution contrefactuelle={ {y: round(p, 6) for y, p in normalized.items()} }."
        )
        return normalized, evidence_mass

    def get_cot(self,*,target_state: Any = 1,distribution: Optional[Mapping[Any, float]] = None,evidence_mass: Optional[float] = None,n_round: int = 6,) -> str:
        """Raisonnement pas-a-pas pour la requete L2.25 stockee sur le SWIG.
    
        Si `distribution` et `evidence_mass` sont fournis, on les reutilise.
        Cela permet a compute_cbn_225 de produire la distribution et la trace
        sans relancer le calcul.
        """
        target_source, evidence = self._stored_225_query()
        if distribution is None or evidence_mass is None:
            distribution, evidence_mass = self._distribution_and_mass(target_source,evidence)
            distribution = self._complete_distribution(target_source, distribution)
    
        # Plus de label scinde : la cible EST la variable source ; le do(...) ci-dessous
        # indique deja qu'on est dans le monde contrefactuel.
        target_label = target_source
        answer = distribution.get(target_state, 0.0)

        do_txt = ", ".join(
            f"{v}={val}" for v, val in sorted(self.intervention.items())
        )
        ev_txt = ", ".join(
            f"{v}={val}" for v, val in sorted(evidence.items())
        ) or "none"
    
        dist_txt = ", ".join(
            f"P({target_label}={s})={p:.{n_round}f}"
            for s, p in sorted(distribution.items(), key=lambda kv: str(kv[0]))
        )
    
        return "\n".join([
            f"Step 1 (Query). Compute P({target_label} = {target_state} | {ev_txt}): the "
            f"probability that {target_source} would equal {target_state} under do({do_txt}), "
            f"given the factual observation(s) [{ev_txt}].",
    
            "Step 2 (Response-function representation, Balke & Pearl 1994). Each source "
            "mechanism P(V | parents) is represented as an exogenous response variable r_V. "
            "The factual world and the do()-world share the same response functions; "
            "they differ only via the intervention.",
    
            f"Step 3 (Abduction). Restrict to the response-function draws whose factual world "
            f"matches the evidence [{ev_txt}]. Their cumulative mass is "
            f"P(evidence) = {evidence_mass:.{n_round}f}.",
    
            f"Step 4 (Action). Force do({do_txt}) in the counterfactual world.",
    
            f"Step 5 (Prediction). Propagate the same response functions under do(); the "
            f"counterfactual distribution of {target_label} is [{dist_txt}], hence "
            f"P({target_label} = {target_state}) = {answer:.{n_round}f}.",
        ])

    
    def _target_source(self, target: Any) -> str:
        """Verifie que la cible est bien une variable du SCM source (plus de labels
        scindes : la cible est directement une variable source)."""
        t = str(target)
        if t in self.source_nodes:
            return t
        raise ValueError(f"Cible inconnue : {target!r}.")

    def _normalize_factual_evidence(self, evidence: Mapping[Any, Any]) -> Dict[str, Any]:
        """Normalise les cles de l'evidence vers des variables source."""
        normalized: Dict[str, Any] = {}
        for key, value in evidence.items():
            source = self._target_source(key)
            normalized[source] = value
        return normalized

    def _stored_225_query(self) -> Tuple[str, Dict[str, Any]]:
        """Return the L2.25 query stored on the SWIG.
    
        The SWIG is responsible for generating and storing the query.
        The engine is responsible for computing it.
        """
        if self.swig.target is None:
            raise ValueError(
                "Aucune requete L2.25 stockee : appeler swig.generate_225() d'abord."
            )
    
        target_source = self._target_source(self.swig.target)
        evidence = self._normalize_factual_evidence(self.swig.observations or {})
        return target_source, evidence

    def query(self) -> Dict[Any, float]:
        """Return only the distribution for the stored L2.25 query."""
        distribution, _ = self.compute_cbn_225()
        return distribution

    def answer(self, *, target_state: Any = 1) -> float:
        """Return only P(target = target_state) for the stored L2.25 query."""
        distribution, _ = self.compute_cbn_225(target_state=target_state)
        return distribution.get(target_state, 0.0)

    def _complete_distribution(self, target_source: str, distribution: Mapping[Any, float]) -> Dict[Any, float]:
        """Add missing target states with probability 0.0."""
        return {
            state: distribution.get(state, 0.0)
            for state in self.mechanisms[target_source].states
        }


    @staticmethod
    def _round_distribution(distribution: Mapping[Any, float],n_round: Optional[int],) -> Dict[Any, float]:
        """Round probabilities only when requested."""
        if n_round is None:
            return dict(distribution)
        return {
            state: round(probability, n_round)
            for state, probability in distribution.items()
        }
    def compute_cbn_225(self,*,n_round: Optional[int] = None,target_state: Any = 1,) -> Tuple[Dict[Any, float], str]:
        """Compute the stored CBN2.25 query.
        The query must have been prepared by `swig.generate_225()`.
        Returns
        -------
        tuple
            (distribution, cot)
        """
        target_source, evidence = self._stored_225_query()
    
        distribution, evidence_mass = self._distribution_and_mass(
            target_source,
            evidence
        )
    
        distribution = self._complete_distribution(target_source, distribution)
        distribution = self._round_distribution(distribution, n_round)
    
        cot = self.get_cot(
            target_state=target_state,
            distribution=distribution,
            evidence_mass=evidence_mass,
            n_round=n_round if n_round is not None else 6,
        )
    
        return distribution, cot

# to_nl_DBN (description NL du SCM source) s'appuie sur la to_nl des CPDs.
TabularCPD.to_nl = to_nl_CPD


__all__ = [
    "Swig",
    "SwigConfig",
    "SwigCounterfactualEngine",
]

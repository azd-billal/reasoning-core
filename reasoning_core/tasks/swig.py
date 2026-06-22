from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import networkx as nx
import numpy as np

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



class Swig(DiscreteBayesianNetwork):
    """Single-World Intervention Graph (Richardson & Robins, 2013).

    Le reseau bayesien causal source est tire DIRECTEMENT (DAG + CPDs) par pgmpy
    `DiscreteBayesianNetwork.get_random`, puis transforme par node-splitting :
    chaque variable d'intervention X est scindee en un noeud FIXE `do(X=v)` et un
    noeud ALEATOIRE indexe `Y(x)`.

    Repartition des roles (volontaire) :
      - le **graphe scinde** materialise dans `self` (+ `SwigSpec`) est la
        REPRESENTATION du monde contrefactuel, consommee par `to_nl` pour produire
        l'enonce des taches ;
      - le **SCM source** (`self.source_bn`, avec ses CPDs) porte les mecanismes ;
      - le **calcul numerique** des contrefactuels n'est PAS dans cette classe : il
        est fait par `SwigCounterfactualEngine`, qui lit `source_bn` + `spec`.

    Surface : `generate_random_swig` / `generate_225` (tirage + node-splitting) et
    `to_nl` (description en langage naturel).
    """

    def __init__(
            self,
            config: Optional[SwigConfig] = None,
            *,
            ebunch: Optional[Iterable[Tuple[str, str]]] = None,
            spec: Optional[SwigSpec] = None,
            bn: Optional[DiscreteBayesianNetwork] = None,
    ):
        super().__init__(ebunch)
        self.config = config if config is not None else SwigConfig()
        self._rng = random.Random(self.config.random_seed)
        self.spec = spec if spec is not None else SwigSpec()
        self.latents: Set[str] = set()
        # BN source (SCM). Peut etre fourni au constructeur (`bn=`) pour imposer le
        # reseau voulu ; sinon rempli par generate_random_swig (tirage aleatoire).
        self.source_bn: Optional[DiscreteBayesianNetwork] = bn
        # Requete L2.25 stockee par generate_225 (au lieu d'etre renvoyee).
        self.target: Optional[str] = None
        self.observations: Optional[Dict[str, Any]] = None
        self.situation_initiale: Optional[Dict[str, Any]] = None
        self.swig_description: Optional[str] = None

    def generate_random_swig(
            self,
            nb_nodes: Optional[int] = None,
            prob_link: Optional[float] = None,
            *,
            interventions: Optional[Mapping[str, Any]] = None,
            node_names: Optional[List[str]] = None,
            num_latents: Optional[int] = None,
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
        3. Node-splitting (SWIG-1.pdf, section 3) puis pose des CPDs sur le graphe
           scinde (memes valeurs que le BN source, cf. `_build_swig_cpds`).

        Un SWIG est toujours defini par une intervention (jamais de "mode source").

        Parametres (defauts dans `self.config`) : `nb_nodes`, `prob_link`
        (densite d'aretes), `node_names` (sa longueur prime sur nb_nodes),
        `num_latents`, `cardinality`, `bn` (BN source impose, p.ex. pour tests).
        """
        if nb_nodes is not None:
            self.config.num_nodes = nb_nodes
        if prob_link is not None:
            self.config.graph_density = prob_link
        if num_latents is not None:
            self.config.num_latents = num_latents
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
        return self._node_split(bn, interventions)

    def _random_intervention(self, bn: DiscreteBayesianNetwork) -> Dict[str, Any]:
        """Choisit au hasard une intervention do(X=v) valide : X doit avoir au moins
        un descendant (sinon la requete contrefactuelle serait triviale)."""
        graph = nx.DiGraph()
        graph.add_nodes_from(str(n) for n in bn.nodes())
        graph.add_edges_from((str(u), str(v)) for u, v in bn.edges())
        candidates = sorted(n for n in graph.nodes() if graph.out_degree(n) > 0)
        if not candidates:
            raise ValueError("Aucune variable avec descendant : SWIG trivial.")
        x_var = self._rng.choice(candidates)
        x_states = list(bn.get_cpds(x_var).state_names[x_var])
        return {x_var: self._rng.choice(x_states)}

    def _node_split(
            self,
            bn: DiscreteBayesianNetwork,
            interventions: Mapping[str, Any],
            *,
            num_latents: Optional[int] = None,
    ) -> "Swig":
        """Node-splitting du reseau source `bn`
        sous `interventions` : remplit `self.source_bn`, `self.spec`, `self.latents`,
        materialise le graphe scinde dans `self` et y pose les CPDs. Appele par
        generate_random_swig. `num_latents` (sinon `self.config.num_latents`) permet
        de forcer le nombre de latentes SANS muter la config."""
        # Repartir d'un graphe propre : permet de re-scinder le MEME objet sous une
        # autre intervention (p.ex. generate_225(intervention=...) appele plusieurs
        # fois), sans accumuler les noeuds/CPDs de la scission precedente.
        existing_cpds = self.get_cpds()
        if existing_cpds:
            self.remove_cpds(*existing_cpds)
        if self.nodes():
            self.remove_nodes_from(list(self.nodes()))
        self.source_bn = bn
        source_nodes = tuple(str(n) for n in bn.nodes())
        source_edges = tuple((str(u), str(v)) for u, v in bn.edges())

        requested = self.config.num_latents if num_latents is None else num_latents
        n_latents = min(requested, max(0, len(source_nodes) - 1))
        if n_latents > 0:
            ordered = sorted(source_nodes)
            confounders = [n for n in ordered if bn.out_degree(n) >= 2]
            others = [n for n in ordered if bn.out_degree(n) < 2]
            n_conf = min(n_latents, len(confounders))
            source_latents = set(self._rng.sample(confounders, n_conf))
            source_latents.update(self._rng.sample(others, n_latents - n_conf))
        else:
            source_latents = set()

        normalized = {str(var): value for var, value in dict(interventions).items()}
        missing = set(normalized) - set(source_nodes)
        if missing:
            raise ValueError(f"Unknown intervention variables: {sorted(missing)}")

        fixed_of = {var: f"do({var}={value})" for var, value in normalized.items()}

        temp = nx.DiGraph()
        temp.add_nodes_from(source_nodes)
        temp.add_nodes_from(fixed_of.values())
        for parent, child in source_edges:
            temp.add_edge(fixed_of[parent] if parent in fixed_of else parent, child)

        fixed_labels = set(fixed_of.values())
        fixed_var_by_label = {label: var for var, label in fixed_of.items()}
        random_of: Dict[str, str] = {}
        node_info: Dict[str, SwigNodeInfo] = {}
        source_of: Dict[str, str] = {}

        for source_var in source_nodes:
            fixed_anc_labels = sorted(nx.ancestors(temp, source_var) & fixed_labels)
            fixed_anc_vars = tuple(fixed_var_by_label[label] for label in fixed_anc_labels)
            if fixed_anc_vars:
                suffix = ",".join(f"{a}={normalized[a]}" for a in sorted(fixed_anc_vars))
                label = f"{source_var}({suffix})"
            else:
                label = source_var
            random_of[source_var] = label
            source_of[label] = source_var
            node_info[label] = SwigNodeInfo(
                label=label, source=source_var, kind="random",
                fixed_ancestors=fixed_anc_vars,
            )
        for source_var, label in fixed_of.items():
            source_of[label] = source_var
            node_info[label] = SwigNodeInfo(
                label=label, source=source_var, kind="fixed",
                value=normalized[source_var],
            )
        if len(node_info) != len(source_nodes) + len(fixed_of):
            raise ValueError("SWIG labels are not unique.")

        self.spec = SwigSpec(
            source_nodes=source_nodes, source_edges=source_edges,
            interventions=normalized, random_of=random_of,
            fixed_of=fixed_of, source_of=source_of, node_info=node_info,
        )
        self.add_nodes_from(node_info)
        for parent, child in source_edges:
            swig_parent = fixed_of[parent] if parent in fixed_of else random_of[parent]
            self.add_edge(swig_parent, random_of[child])
        # Les latentes du DAG source restent latentes dans le SWIG, representees
        # par leur moitie aleatoire (un noeud du SWIG).
        self.latents = {random_of[n] for n in source_latents if n in random_of}
        self._build_swig_cpds()
        return self

    def _build_swig_cpds(self) -> None:
        """Pose les CPDs sur le graphe scinde `self` pour en faire un reseau
        bayesien complet (ce qui justifie l'heritage DiscreteBayesianNetwork).

        Le node-splitting SEPARE chaque variable, il n'en cree pas : les VALEURS des
        CPDs sont donc IDENTIQUES a celles du BN source (jamais recalculees) :
          - noeud aleatoire (moitie naturelle, ex. `X` ou un enfant `Y(X=v)`) : meme
            CPD que dans le BN source. Si le label est inchange (`label == source`) on
            REUTILISE la CPD telle quelle ; si le noeud est renomme (descendant de
            l'intervention) on reconstruit la TabularCPD avec les parents REETIQUETES
            vers les noeuds du SWIG, en gardant les MEMES valeurs (pgmpy ne sait pas
            renommer une CPD existante) ;
          - noeud fixe `do(X=v)` (l'autre moitie, l'intervention) : masse-point
            P(do(X=v)=v)=1. Ce n'est pas une CPD du BN mais l'encodage de "cette
            moitie vaut v" -- l'intervention elle-meme. Ce noeud est absent du BN
            source (cree par la scission), donc construit ici."""
        cpds: List[TabularCPD] = []
        for src, label in self.spec.fixed_of.items():
            states = list(self.source_bn.get_cpds(src).state_names[src])
            col = np.zeros((len(states), 1))
            col[states.index(self.spec.interventions[src]), 0] = 1.0
            cpds.append(TabularCPD(label, len(states), col, state_names={label: states}))
        for src in self.spec.source_nodes:
            cpd = self.source_bn.get_cpds(src)
            label = self.spec.random_of[src]
            if label == src:                       # rien renomme -> CPD source reutilisee
                cpds.append(cpd.copy())
                continue
            src_par = [str(p) for p in cpd.variables[1:]]
            swig_par = [self.spec.fixed_of.get(p, self.spec.random_of[p]) for p in src_par]
            state_names = {label: list(cpd.state_names[cpd.variable])}
            for sp, p in zip(swig_par, src_par):
                state_names[sp] = list(cpd.state_names[p])
            kwargs: Dict[str, Any] = dict(
                variable=label, variable_card=int(cpd.variable_card),
                values=cpd.get_values(), state_names=state_names,
            )
            if swig_par:
                kwargs["evidence"] = swig_par
                kwargs["evidence_card"] = [int(c) for c in cpd.cardinality[1:]]
            cpds.append(TabularCPD(**kwargs))
        self.add_cpds(*cpds)
        if not self.check_model():
            raise RuntimeError("CPDs du SWIG mal formees.")

    def generate_225(
            self,
            *,
            intervention: Optional[Mapping[str, Any]] = None,
            target: Optional[Any] = None,
            observations: Optional[Mapping[str, Any]] = None,
    ) -> "Swig":
        """Construit une requete L2.25 sur ce SWIG et la STOCKE sur l'objet
        (`self.target`, `self.observations`, `self.situation_initiale`,
        `self.swig_description`), au lieu de renvoyer un dico. Renvoie `self`
        (chainage). La REPONSE se calcule a part avec
        `SwigCounterfactualEngine(self).compute_cbn_225(self.target, self.observations)`.

        Une intervention `{X = x}` definit le monde contrefactuel : par la condition
        (ii) de la Def. 11, la MEME valeur `x` se propage a TOUS les descendants de X
        -- exactement le node-splitting du SWIG.

        L2.25 STRICT : la requete ne conditionne QUE sur des variables NON descendantes
        de l'intervention (Def. 11, Ex. 6). Conditionner sur la valeur factuelle d'un
        descendant de X donnerait une jointe inter-mondes (p.ex. P(Y_x, M)) = L3, hors
        L2.25. La valeur factuelle de X lui-meme reste permise.

        Parametres (tous optionnels) :
          - `intervention` : si fournie, (re)scinde le BN source `self.source_bn` sous
            cette intervention ; sinon on reutilise le SWIG deja construit (p.ex. par
            `generate_random_swig`).
          - `target`       : noeud contrefactuel interroge (variable source OU noeud
            SWIG) ; tire au hasard parmi les descendants contrefactuels si omis.
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
            self._node_split(self.source_bn, intervention)
        if not self.spec.interventions:
            raise ValueError("generate_225 exige un SWIG construit avec intervention(s).")

        if target is None:
            # cibles candidates = noeuds aleatoires touches par l'intervention
            # (ceux qui ont un ancetre fixe), i.e. les descendants contrefactuels.
            candidates = sorted(
                lbl for lbl, info in self.spec.node_info.items()
                if info.is_random and info.fixed_ancestors
            )
            if not candidates:
                raise ValueError("Aucun descendant contrefactuel a interroger.")
            target_label = self._rng.choice(candidates)
        else:
            src = self.spec.source_of.get(str(target), str(target))
            target_label = self.spec.random_of[src]
        target_source = self.spec.source_of[target_label]

        # L2.25 STRICT (Def. 11 + Ex. 6 du papier) : on ne conditionne QUE sur des
        # variables NON descendantes de l'intervention. La jointe L2.25 indexee par
        # {X=x} est P(X, Z_x, Y_x) -- X factuel, descendants tous indices x. Conditionner
        # sur la valeur FACTUELLE d'un descendant de X melerait deux submodels (p.ex.
        # P(Y_x, M)), ce qui est du L3, hors L2.25. La valeur factuelle de X lui-meme
        # reste permise (non indicee dans la jointe).
        src_graph = nx.DiGraph()
        src_graph.add_nodes_from(self.spec.source_nodes)
        src_graph.add_edges_from(self.spec.source_edges)
        post_treatment: Set[str] = set()
        for x_var in self.spec.interventions:
            post_treatment |= nx.descendants(src_graph, x_var)

        if observations is None:
            sample = self.source_bn.simulate(n_samples=1, show_progress=False).iloc[0]
            obs: Dict[str, Any] = {}
            for var in self.spec.source_nodes:
                if var == target_source or var in post_treatment or var not in sample.index:
                    continue
                value = sample[var]
                obs[var] = value.item() if hasattr(value, "item") else value
        else:
            obs = {}
            for key, value in observations.items():
                src = self.spec.source_of.get(str(key), str(key))
                if src in post_treatment:
                    raise ValueError(
                        f"Observation L2.25 invalide : {key!r} est un descendant de "
                        f"l'intervention. Conditionner sur sa valeur factuelle donne une "
                        f"requete inter-mondes (L3), hors L2.25."
                    )
                obs[str(key)] = value

        # On STOCKE la requete sur le SWIG plutot que de la renvoyer.
        self.target = target_label
        self.observations = obs
        self.situation_initiale = dict(self.spec.interventions)
        self.swig_description = self.to_nl()
        return self

    def copy(self) -> "Swig":
        """Copie de la STRUCTURE du SWIG (graphe scinde + CPDs + `SwigSpec` +
        latentes). Le SCM source `source_bn` est PARTAGE (ses CPDs sont en lecture
        seule pour le moteur contrefactuel), seul le `spec` est recopie en profondeur."""
        copied = Swig(
            config=copy.deepcopy(self.config),
            spec=copy.deepcopy(self.spec),
            bn=self.source_bn,  # SCM source partage (CPDs en lecture seule)
        )
        copied.add_nodes_from(self.nodes())
        copied.add_edges_from(self.edges())
        copied.latents = set(self.latents)
        for cpd in self.get_cpds():        # CPDs posees sur le SWIG, recopiees
            copied.add_cpds(cpd.copy())
        # Requete L2.25 stockee (generate_225), recopiee si presente.
        copied.target = self.target
        copied.observations = dict(self.observations) if self.observations is not None else None
        copied.situation_initiale = (
            dict(self.situation_initiale) if self.situation_initiale is not None else None
        )
        copied.swig_description = self.swig_description
        return copied


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
        if not swig.spec.interventions:
            raise ValueError("Le moteur exige un SWIG construit avec intervention(s).")
        if swig.latents or getattr(swig.source_bn, "latents", set()):
            raise ValueError("SCM markovien requis : pas de variable latente.")
        self.swig = swig
        self.source_bn = swig.source_bn
        self.spec = swig.spec
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
        do_txt = ", ".join(f"{v}={x}" for v, x in sorted(self.spec.interventions.items()))
        self.trace.append(
            f"Phase 1 (representation) : modele a fonctions de reponse construit sur "
            f"{len(self.mechanisms)} mecanismes ; intervention do({do_txt})."
        )

    def _build_response_functions(self) -> Dict[str, _Mechanism]:
        mech: Dict[str, _Mechanism] = {}
        for var in self.spec.source_nodes:
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
        graph = nx.DiGraph()
        graph.add_nodes_from(self.spec.source_nodes)
        graph.add_edges_from(self.spec.source_edges)
        return list(nx.topological_sort(graph))

    def _prob(self, var: str, value: Any, parent_config: Mapping[str, Any]) -> float:
        """P(var = value | parents = parent_config), lue dans la CPD source."""
        assignment = {var: value, **parent_config}
        return float(self.mechanisms[var].cpd.get_value(**assignment))

    def _response_space_size(self) -> int:
        """Nombre de jeux de tables de reponse completes enumeres : produit, par
        noeud, du nombre de fonctions de reponse `k^(#configurations de parents)`."""
        size = 1
        for var in self.spec.source_nodes:
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

    def _distribution_and_mass(
            self, target_source: str, evidence: Mapping[str, Any]
    ) -> Tuple[Dict[Any, float], float]:
        """Coeur des phases 2-3 : renvoie (distribution contrefactuelle normalisee
        de la cible, masse d'evidence P(evidence)). Conserve pour `get_cot`, seul
        consommateur qui a besoin de la masse ; `query`/`answer` passent desormais
        par `compute_cbn_225`. Meme machinerie de tables de reponse completes que
        `compute_cbn_225` (garde-fou dans _iter_response_tables)."""
        interventions = dict(self.spec.interventions)
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

    def query(
            self, target: Any, factual_evidence: Optional[Mapping[Any, Any]] = None
    ) -> Dict[Any, float]:
        """P(cible* | evidence factuelle) sous l'intervention du SWIG : distribution
        {valeur: probabilite} normalisee. `target` et les cles de `factual_evidence`
        peuvent etre des variables source ou des noeuds SWIG."""
        # Option A : meme calcul que compute_cbn_225, on le reutilise directement.
        # _distribution_and_mass n'est garde que pour get_cot (besoin de P(evidence)).
        return self.compute_cbn_225(target, factual_evidence)

    def answer(
            self,
            target: Any,
            factual_evidence: Optional[Mapping[Any, Any]] = None,
            *,
            target_state: Any = 1,
    ) -> float:
        """REPONSE scalaire : P(cible* = `target_state` | evidence factuelle) sous
        l'intervention du SWIG (la quantite a comparer dans une tache)."""
        return self.compute_cbn_225(target, factual_evidence).get(target_state, 0.0)

    def get_cot(
            self,
            target: Any,
            factual_evidence: Optional[Mapping[Any, Any]] = None,
            *,
            target_state: Any = 1,
    ) -> str:
        """Raisonnement pas-a-pas (chain-of-thought) calque sur l'algorithme du
        papier (3 phases abduction-action-prediction) menant a la reponse."""
        target_source = self._target_source(target)
        evidence = self._normalize_factual_evidence(factual_evidence or {})
        distribution, mass = self._distribution_and_mass(target_source, evidence)
        answer = distribution.get(target_state, 0.0)
        target_label = self.spec.random_of.get(target_source, target_source)
        do_txt = ", ".join(f"{v}={val}" for v, val in sorted(self.spec.interventions.items()))
        ev_txt = ", ".join(f"{v}={val}" for v, val in sorted(evidence.items())) or "none"
        dist_txt = ", ".join(
            f"P({target_label}={s})={p:.6f}"
            for s, p in sorted(distribution.items(), key=lambda kv: str(kv[0]))
        )
        return "\n".join([
            f"Step 1 (Query). Compute P({target_label} = {target_state} | {ev_txt}): the "
            f"probability that {target_source} would equal {target_state} under do({do_txt}), "
            f"given the factual observation(s) [{ev_txt}].",
            "Step 2 (Response-function representation, Balke & Pearl 1994). Each source "
            "mechanism P(V | parents) is recast as an exogenous response variable r_V (it picks "
            "V's value for every parent configuration), shared by the factual world and the "
            "do()-world; the two worlds differ only via the intervention. The counterfactual is "
            "evaluated by exact enumeration over these shared response functions -- the functional "
            "equivalent of the twin network, without building an explicit twin graph.",
            f"Step 3 (Abduction). Restrict to the response-function draws whose FACTUAL world "
            f"matches the evidence [{ev_txt}]; their cumulative mass is P(evidence) = {mass:.6f}, "
            f"which defines the posterior over response functions.",
            f"Step 4 (Action). Force do({do_txt}) in the counterfactual world (sever the "
            f"incoming edges of the intervened variable(s)).",
            f"Step 5 (Prediction). Propagate the SAME response functions under do(); the "
            f"counterfactual distribution of {target_label} is [{dist_txt}], hence "
            f"P({target_label} = {target_state}) = {answer:.6f}.",
        ])

    def compute_cbn_225(self, target, factual_evidence=None, *, n_round=None):
        evidence = self._normalize_factual_evidence(factual_evidence or {})
        target_source = self._target_source(target)

        weights = {state: 0.0 for state in self.mechanisms[target_source].states}
        evidence_weight = 0.0

        for response_tables, response_weight in self._iter_response_tables():
            # 1. Abduction: evaluate the factual world
            factual_world = self._evaluate_world(
                response_tables,
                interventions={}
            )

            # Keep only worlds compatible with factual evidence
            if not self._world_matches(factual_world, evidence):
                continue

            evidence_weight += response_weight

            # 2. Action: apply the SWIG intervention
            counterfactual_world = self._evaluate_world(
                response_tables,
                interventions=self.swig.spec.interventions
            )

            # 3. Prediction: accumulate the target value
            target_value = counterfactual_world[target_source]
            weights[target_value] += response_weight
        if evidence_weight == 0:
            raise ValueError(
                "The factual evidence has probability zero under the model."
            )
        distribution = {
            state: weight / evidence_weight
            for state, weight in weights.items()
        }
        if n_round is not None:
            distribution = {
                state: round(probability, n_round)
                for state, probability in distribution.items()
            }
        return distribution

    def _target_source(self, target: Any) -> str:
        """Resout une cible (variable source OU noeud SWIG) vers sa variable source."""
        t = str(target)
        if t in self.spec.source_of:      # noeud SWIG -> variable source
            return self.spec.source_of[t]
        if t in self.spec.source_nodes:   # deja une variable source
            return t
        raise ValueError(f"Cible inconnue : {target!r}.")

    def _normalize_factual_evidence(self, evidence: Mapping[Any, Any]) -> Dict[str, Any]:
        """Normalise les cles de l'evidence vers des variables source."""
        normalized: Dict[str, Any] = {}
        for key, value in evidence.items():
            source = self._target_source(key)
            normalized[source] = value
        return normalized

Swig.to_nl = to_nl_DBN
TabularCPD.to_nl = to_nl_CPD


__all__ = [
    "Swig",
    "SwigConfig",
    "SwigSpec",
    "SwigNodeInfo",
    "SwigCounterfactualEngine",
]
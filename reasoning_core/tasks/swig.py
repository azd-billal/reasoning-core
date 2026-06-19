from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork

from reasoning_core.template import Config


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
    """Single-World Intervention Graph (Richardson & Robins.

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
    `to_nl` (description en langage naturel)..
    """

    def __init__(
            self,
            config: Optional[SwigConfig] = None,
            *,
            ebunch: Optional[Iterable[Tuple[str, str]]] = None,
            spec: Optional[SwigSpec] = None,
            bn_225: Optional[DiscreteBayesianNetwork] = None,
    ):
        super().__init__(ebunch)
        self.config = config if config is not None else SwigConfig()
        self._rng = random.Random(self.config.random_seed)
        self.spec = spec if spec is not None else SwigSpec()
        self.latents: Set[str] = set()

        self.source_bn: Optional[DiscreteBayesianNetwork] = None

        self.bn_225 = bn_225

    def generate_random_swig(
            self,
            nb_nodes: Optional[int] = None,
            prob_link: Optional[float] = None,
            *,
            interventions: Mapping[str, Any],
            node_names: Optional[List[str]] = None,
            num_latents: Optional[int] = None,
            cardinality: Optional[int] = None,
    ) -> "Swig":
        """Tire un SWIG aleatoire de bout en bout.

        1. pgmpy `DiscreteBayesianNetwork.get_random` construit DIRECTEMENT un
           reseau bayesien causal source (DAG + CPDs) ; on en retient la
           structure (cardinalite fixee a `config.cardinality`).
        2. Node-splitting STRUCTUREL (SWIG-1.pdf, section 3) : chaque variable
           d'intervention X est scindee en un noeud fixe `do(X=v)` et un noeud
           aleatoire indexe `Y(x)`. On remplit le `SwigSpec` et on materialise le
           graphe scinde dans `self`.

        `interventions` (do(X=v)) est **obligatoire** : un SWIG est defini par son
        intervention (sans elle il n'y a pas de scission, juste le DAG source).

        Parametres (defauts dans `self.config`) : `nb_nodes`, `prob_link`
        (densite d'aretes), `node_names` (sa longueur prime sur nb_nodes),
        `num_latents`, `cardinality`.
        """
        if not interventions:
            raise ValueError(
                "generate_random_swig exige au moins une intervention do(X=v) : "
                "un SWIG est defini par son intervention."
            )
        if nb_nodes is not None:
            self.config.num_nodes = nb_nodes
        if prob_link is not None:
            self.config.graph_density = prob_link
        if num_latents is not None:
            self.config.num_latents = num_latents
        if cardinality is not None:
            self.config.cardinality = cardinality

        k = self.config.cardinality
        n_nodes = len(node_names) if node_names is not None else self.config.num_nodes

        bn = DiscreteBayesianNetwork.get_random(
            n_nodes=n_nodes,
            edge_prob=self.config.graph_density,
            node_names=[str(x) for x in node_names] if node_names is not None else None,
            n_states=k,
            seed=self._rng.randrange(2**31),
        )
        return self._node_split(bn, interventions)

    def _node_split(
        self,
        bn: DiscreteBayesianNetwork,
        interventions: Mapping[str, Any],
        *,
        num_latents: Optional[int] = None,
    ) -> "Swig":
        """Node-splitting du reseau source `bn`
        sous `interventions` : remplit `self.source_bn`, `self.spec`, `self.latents`
        et materialise le graphe scinde dans `self`. Partage par generate_random_swig
        et generate_225. `num_latents` (sinon `self.config.num_latents`) permet de
        forcer le nombre de latentes SANS muter la config."""
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
        return self

    def generate_225(
        self,
        *,
        intervention: Optional[Mapping[str, Any]] = None,
        target: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Genere une requete L2.25

        Une intervention `{X = x}` definit le monde contrefactuel : par la
        condition (ii) de la Def. 11, la MEME valeur `x` se propage a TOUS les
        descendants de X -- ce qui est exactement le node-splitting du SWIG. On
        observe ensuite un jeu de valeurs factuelles, et la requete porte sur un
        descendant contrefactuel.

        Renvoie le dico decrivant la requete (la REPONSE se calcule a part avec
        `SwigCounterfactualEngine(swig).compute_cbn_225(target, observations)`) :
          - `situation_initiale` : l'intervention `{X = x}` (l'antecedent L2.25) ;
          - `observations`       : les valeurs factuelles observees `{Z=z, A=a, ...}` ;
          - `target`             : le noeud contrefactuel interroge (ex. `Y(X=x)`) ;
          - `swig_description`   : la description en langage naturel du SWIG.
        """

        if self.bn_225 is not None:
            bn = self.bn_225
        else:
            bn = DiscreteBayesianNetwork.get_random(
        # pour ne pas briser l'invariant markovien des configs sans latente.
                n_nodes=self.config.num_nodes,
                edge_prob=self.config.graph_density,
                n_states=self.config.cardinality,
                seed=self._rng.randrange(2**31),
            )

        graph = nx.DiGraph()
        graph.add_nodes_from(str(n) for n in bn.nodes())
        graph.add_edges_from((str(u), str(v)) for u, v in bn.edges())

        if intervention is None:
            candidates = sorted(n for n in graph.nodes() if graph.out_degree(n) > 0)
            if not candidates:
                raise ValueError("Aucune variable avec descendant : requete L2.25 triviale.")
            x_var = self._rng.choice(candidates)
            x_states = list(bn.get_cpds(x_var).state_names[x_var])
            intervention = {x_var: self._rng.choice(x_states)}
        intervened = {str(v) for v in intervention}

        self._node_split(bn, intervention, num_latents=0)

        if target is None:
            descendants = sorted(
                {d for v in intervened for d in nx.descendants(graph, v)} - intervened
            )
            if not descendants:
                raise ValueError(
                    f"Les variables d'intervention {sorted(intervened)} n'ont pas de descendant."
                )
            target_source = self._rng.choice(descendants)
        else:
            target_source = self.spec.source_of.get(str(target), str(target))
        target_label = self.spec.random_of[target_source]

        sample = bn.simulate(n_samples=1, show_progress=False).iloc[0]
        observations: Dict[str, Any] = {}
        for var in self.spec.source_nodes:
            if var == target_source or var not in sample.index:
                continue
            value = sample[var]
            observations[var] = value.item() if hasattr(value, "item") else value

        return {
            "situation_initiale": dict(intervention),
            "observations": observations,
            "target": target_label,
            "swig_description": self.to_nl(),
        }

    def to_nl(self) -> str:
        # Un SWIG est toujours construit avec une intervention (generate_random_swig
        # / generate_225 l'exigent) : le spec est donc rempli. Garde defensif si on
        # appelle to_nl sur un Swig non genere.
        if not self.spec.node_info:
            raise ValueError(
                "SWIG non construit : appeler generate_random_swig() ou generate_225() "
                "avant to_nl()."
            )
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
        # Mecanismes du SCM source (tables de probabilite conditionnelle) : requis
        # pour calculer numeriquement le contrefactuel.
        if self.source_bn is not None:
            lines.append("Conditional probability tables (source mechanisms):")
            for node in sorted(map(str, self.source_bn.nodes())):
                cpd = self.source_bn.get_cpds(node)
                var = cpd.variable
                evidence = list(cpd.variables[1:])
                states = cpd.state_names[var]
                configs = (
                    list(itertools.product(*[cpd.state_names[e] for e in evidence]))
                    if evidence else [()]
                )
                for combo in configs:
                    assign = dict(zip(evidence, combo))
                    cond = ", ".join(f"{e}={v}" for e, v in assign.items())
                    head = f" | {cond}" if cond else ""
                    terms = ", ".join(
                        f"P({var}={s}{head})={float(cpd.get_value(**{var: s, **assign})):.3f}"
                        for s in states
                    )
                    lines.append(f"- {terms}")
        return "\n".join(lines)

    def copy(self) -> "Swig":
        """Copie de la STRUCTURE du SWIG (graphe scinde + `SwigSpec` + latentes).
        Le SCM source `source_bn` est PARTAGE (ses CPDs sont en lecture seule pour
        le moteur contrefactuel), seul le `spec` est recopie en profondeur."""
        copied = Swig(
            config=copy.deepcopy(self.config),
            spec=copy.deepcopy(self.spec),
        )
        copied.add_nodes_from(self.nodes())
        copied.add_edges_from(self.edges())
        copied.latents = set(self.latents)
        copied.source_bn = self.source_bn  # SCM source partage (CPDs en lecture seule)
        copied.bn_225 = self.bn_225        # entree L2.25 imposee, conservee
        return copied


@dataclass
class _Mechanism:
    """Mecanisme P(V | parents) du SCM source, vu comme fonction de reponse :
    etats de V, parents, configurations de parents (ordre fixe, pour les fonctions
    de reponse completes) et CPD source."""
    source: str
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
    CONTREFACTUEL (le SWIG, sous do(X=x)) PARTAGENT ces `r_V` : c'est le reseau
    jumeau de la Figure 2. On evalue exactement P(cible* | evidence) en enumerant
    les `r_V` aux seules configurations utiles (factuelle + contrefactuelle).

    Les trois phases du papier (algorithme p.234) :
      1. ABDUCTION  : conditionner les `r_V` sur l'evidence factuelle observee ;
      2. ACTION     : forcer do(X=x) dans le monde contrefactuel (deja encode par
                      le node-splitting du SWIG) ;
      3. PREDICTION : propager les memes `r_V` dans le monde do() et lire la cible.
    Abduction et prediction se calculent en une passe (la propagation jumelle de
    l'etape 4 du papier). Prior sur les `r_V` : CANONIQUE, derive des CPDs du SCM
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
                source=var,
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
        """Genere (response_tables, poids) : response_tables = {noeud: table_de_reponse}
        tire sur le produit des fonctions de reponse de chaque noeud, poids = produit
        des priors. Garde-fou commun (protege aussi compute_cbn_225) : refuse une
        enumeration plus grande que `max_response_functions`."""
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

        yield from rec(0, {}, 1.0)

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
        de la cible, masse d'evidence P(evidence)). Meme machinerie de tables de
        reponse completes que compute_cbn_225 (garde-fou dans _iter_response_tables)."""
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
        target_source = self._target_source(target)
        evidence = self._normalize_factual_evidence(factual_evidence or {})
        distribution, _ = self._distribution_and_mass(target_source, evidence)
        return distribution

    def answer(
        self,
        target: Any,
        factual_evidence: Optional[Mapping[Any, Any]] = None,
        *,
        target_state: Any = 1,
    ) -> float:
        """REPONSE scalaire : P(cible* = `target_state` | evidence factuelle) sous
        l'intervention du SWIG (la quantite a comparer dans une tache)."""
        target_source = self._target_source(target)
        evidence = self._normalize_factual_evidence(factual_evidence or {})
        distribution, _ = self._distribution_and_mass(target_source, evidence)
        return distribution.get(target_state, 0.0)

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
            "Step 2 (Twin-network / response functions, Balke & Pearl 1994). Each source "
            "mechanism P(V | parents) is recast as an exogenous response variable r_V shared "
            "by the factual world and the do()-world; the worlds differ only via the intervention.",
            f"Step 3 (Abduction). Condition the response variables on the factual evidence "
            f"[{ev_txt}]; this evidence has probability P(evidence) = {mass:.6f}, defining the "
            f"posterior over response functions.",
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


__all__ = [
    "Swig",
    "SwigConfig",
    "SwigSpec",
    "SwigNodeInfo",
    "SwigCounterfactualEngine",
]
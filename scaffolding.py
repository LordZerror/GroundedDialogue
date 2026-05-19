from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Iterable, Literal

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDFS

ScaffoldLevel = Literal["LOW", "MEDIUM", "HIGH"]
Strategy = Literal["SOCRATIC", "CONTRAST", "WORKED_EXAMPLE", "EXPLAIN"]

CSO = Namespace("http://cso.kmi.open.ac.uk/schema/cso#")
SCHEMA = Namespace("http://schema.org/")

SCAFFOLD_DEPTH = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}

SCAFFOLD_MATRIX = {
    "Beginner": {
        "Remember": "HIGH",
        "Understand": "HIGH",
        "Apply": "HIGH",
        "Analyze": "HIGH",
        "Evaluate": "HIGH",
        "Create": "HIGH",
    },
    "Intermediate": {
        "Remember": "MEDIUM",
        "Understand": "MEDIUM",
        "Apply": "MEDIUM",
        "Analyze": "HIGH",
        "Evaluate": "HIGH",
        "Create": "HIGH",
    },
    "Proficient": {
        "Remember": "LOW",
        "Understand": "LOW",
        "Apply": "MEDIUM",
        "Analyze": "MEDIUM",
        "Evaluate": "HIGH",
        "Create": "HIGH",
    },
}


@dataclass(frozen=True)
class ConceptNode:
    uri: str
    label: str


@dataclass(frozen=True)
class ScaffoldContext:
    scaffold_level: ScaffoldLevel
    concept_chain: list[ConceptNode]
    wikipedia_links: list[str]
    grounded: bool


@dataclass(frozen=True)
class ScaffoldResponse:
    strategy: Strategy
    misunderstanding: str
    context: ScaffoldContext
    message: str


class OntologyService:
    def __init__(self, ontology_path: str | Path | None = None, graph: Graph | None = None):
        if graph is None and ontology_path is None:
            ontology_path = Path(__file__).with_name("CSO 3.ttl")

        self.graph = graph if graph is not None else Graph()
        if graph is None:
            self.graph.parse(Path(ontology_path), format="ttl")

        self._label_index_exact: dict[str, URIRef] = {}
        self._label_index_normalized: dict[str, URIRef] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        for subject, label in self.graph.subject_objects(RDFS.label):
            if isinstance(subject, URIRef):
                label_text = str(label)
                self._label_index_exact.setdefault(label_text, subject)
                self._label_index_normalized.setdefault(self._normalize_label(label_text), subject)

    @staticmethod
    def _normalize_label(value: str) -> str:
        lowered = value.strip().lower().replace("_", " ").replace("-", " ")
        return re.sub(r"\s+", " ", lowered)

    def _node_for_uri(self, uri: URIRef) -> ConceptNode:
        label = next(self.graph.objects(uri, RDFS.label), None)
        if label is None:
            label_text = str(uri).rstrip("/").split("/")[-1].replace("_", " ")
        else:
            label_text = str(label)
        return ConceptNode(uri=str(uri), label=label_text)

    def _uri_exists(self, uri: URIRef) -> bool:
        return any(self.graph.triples((uri, None, None))) or any(self.graph.triples((None, None, uri)))

    def _follow_preferential_equivalent(self, uri: URIRef) -> URIRef:
        seen = set()
        current = uri

        while current not in seen:
            seen.add(current)
            preferred = [
                candidate
                for candidate in self.graph.objects(current, CSO.preferentialEquivalent)
                if isinstance(candidate, URIRef)
            ]
            if not preferred:
                break
            current = sorted(preferred, key=str)[0]

        return current

    def _canonicalize_uri(self, uri: URIRef) -> URIRef:
        current = self._follow_preferential_equivalent(uri)
        related_candidates = [
            self._follow_preferential_equivalent(candidate)
            for candidate in self.graph.objects(current, CSO.relatedEquivalent)
            if isinstance(candidate, URIRef)
        ]
        candidates = {current, *related_candidates}
        return min(candidates, key=lambda candidate: (self._node_for_uri(candidate).label.lower(), str(candidate)))

    def resolve_concept(self, label_or_uri: str | ConceptNode | None) -> ConceptNode | None:
        if label_or_uri is None:
            return None

        if isinstance(label_or_uri, ConceptNode):
            return self._node_for_uri(self._canonicalize_uri(URIRef(label_or_uri.uri)))

        raw_value = label_or_uri.strip()
        if not raw_value:
            return None

        subject = None
        if raw_value.startswith("http://") or raw_value.startswith("https://"):
            uri = URIRef(raw_value)
            if self._uri_exists(uri):
                subject = uri

        if subject is None:
            subject = self._label_index_exact.get(raw_value)

        if subject is None:
            subject = self._label_index_normalized.get(self._normalize_label(raw_value))

        if subject is None:
            return None

        return self._node_for_uri(self._canonicalize_uri(subject))

    def _pick_parent(self, uri: URIRef) -> URIRef | None:
        parents = []
        for parent in self.graph.subjects(CSO.superTopicOf, uri):
            if isinstance(parent, URIRef):
                canonical_parent = self._canonicalize_uri(parent)
                if canonical_parent != uri:
                    parents.append(canonical_parent)

        if not parents:
            return None

        return sorted(
            set(parents),
            key=lambda candidate: (self._node_for_uri(candidate).label.lower(), str(candidate)),
        )[0]

    def get_ancestor_chain(self, concept: str | ConceptNode, depth: int) -> list[ConceptNode]:
        resolved = self.resolve_concept(concept)
        if resolved is None:
            return []

        chain = [resolved]
        current_uri = URIRef(resolved.uri)
        seen = {current_uri}

        for _ in range(depth):
            parent_uri = self._pick_parent(current_uri)
            if parent_uri is None or parent_uri in seen:
                break
            seen.add(parent_uri)
            chain.append(self._node_for_uri(parent_uri))
            current_uri = parent_uri

        return chain

    def get_wikipedia_links(self, concept: str | ConceptNode) -> list[str]:
        resolved = self.resolve_concept(concept)
        if resolved is None:
            return []

        wiki_links = []
        for link in self.graph.objects(URIRef(resolved.uri), SCHEMA.relatedLink):
            link_text = str(link)
            if link_text.startswith("http://en.wikipedia.org/") or link_text.startswith("https://en.wikipedia.org/"):
                wiki_links.append(link_text)

        return sorted(set(wiki_links))

    def get_scaffold_context(self, concept: str | ConceptNode | None, scaffold_level: ScaffoldLevel) -> ScaffoldContext:
        resolved = self.resolve_concept(concept)
        if resolved is None:
            return ScaffoldContext(
                scaffold_level=scaffold_level,
                concept_chain=[],
                wikipedia_links=[],
                grounded=False,
            )

        chain = self.get_ancestor_chain(resolved, SCAFFOLD_DEPTH[scaffold_level])
        wiki_links = []
        for node in chain:
            for link in self.get_wikipedia_links(node):
                if link not in wiki_links:
                    wiki_links.append(link)

        return ScaffoldContext(
            scaffold_level=scaffold_level,
            concept_chain=chain,
            wikipedia_links=wiki_links,
            grounded=bool(chain),
        )


class ScaffoldingEngine:
    def __init__(self, ontology_service: OntologyService, llm_chat_fn: Callable[[str, str, str | None], str]):
        self.ontology = ontology_service
        self.chat = llm_chat_fn

    def determine_scaffold_level(self, proficiency: str, bloom_level: str) -> ScaffoldLevel:
        normalized_proficiency = proficiency.title()
        return SCAFFOLD_MATRIX[normalized_proficiency][bloom_level]

    def choose_strategy(self, attempts: int) -> Strategy:
        if attempts <= 1:
            return "SOCRATIC"
        if attempts == 2:
            return "CONTRAST"
        if attempts == 3:
            return "WORKED_EXAMPLE"
        return "EXPLAIN"

    def infer_misunderstanding(self, session, user_input: str) -> str:
        prompt = f"""
Question: {session.current_question}
Correct answer: {session.correct_answer}
Student answer: {user_input}

Infer the likely misunderstanding in one short sentence.
Do not reveal the correct answer.
""".strip()

        return self.chat(
            session.session_id,
            prompt,
            system_prompt="You diagnose student misunderstandings from short tutoring exchanges.",
        )

    def _support_system_prompt(self, strategy: Strategy) -> str:
        prompts = {
            "SOCRATIC": "You MUST NOT provide any explanation or preamble. You MUST ask exactly one focused Socratic question related to the Question that forces the student to rethink their assumption without giving away the answer.",
            "CONTRAST": "You MUST highlight the conceptual difference between the option corresponding to student's answer and the actual mechanism, without giving away the final answer.",
            "WORKED_EXAMPLE": "You MUST provide a brief, analogous scenario or a simplified example that illustrates the underlying concept of correct answer without directly revealing the answer.",
            "EXPLAIN": "You MUST give a clear direct explanation and may reveal the answer.",
        }
        return prompts[strategy]

    @staticmethod
    def _format_concept_chain(concept_chain: Iterable[ConceptNode]) -> str:
        chain = list(concept_chain)
        if not chain:
            return "No ontology concept could be resolved for this question."

        return "\n".join(f"- {node.label} ({node.uri})" for node in chain)

    def generate_support(self, session, user_input: str) -> ScaffoldResponse:
        scaffold_level = self.determine_scaffold_level(session.proficiency, session.bloom_level)
        strategy = self.choose_strategy(session.attempts)
        concept_key = session.question_concept or session.topic
        context = self.ontology.get_scaffold_context(concept_key, scaffold_level)
        misunderstanding = self.infer_misunderstanding(session, user_input)

        prompt = f"""
Teaching strategy: {strategy}
Scaffold level: {scaffold_level}
Topic: {session.topic}
Question concept: {concept_key}
Question: {session.current_question}
Question Options: {session.options}
Correct answer: {session.correct_answer}
Student answer: {user_input}
Likely misunderstanding: {misunderstanding}

Ontology context:
{self._format_concept_chain(context.concept_chain)}

Wikipedia links:
{chr(10).join(context.wikipedia_links) if context.wikipedia_links else "None"}

Write ONE tutor response that is grounded in the ontology context. Follow these strict rules:
1. CRITICAL: NEVER reveal, explain, or heavily hint at the correct answer (unless strategy is EXPLAIN).
2. Use 1-3 sentences maximum.
3. If no ontology concept is provided, give a general conceptual response without pretending the ontology supplied evidence.
4. {self._support_system_prompt(strategy)}
""".strip()

        message = self.chat(
            session.session_id,
            prompt,
            system_prompt=self._support_system_prompt(strategy),
        )

        return ScaffoldResponse(
            strategy=strategy,
            misunderstanding=misunderstanding,
            context=context,
            message=message,
        )

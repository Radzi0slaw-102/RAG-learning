# deterministic triple extraction from dependency parse trees

from dataclasses import dataclass
from typing import Iterator

import spacy
from spacy.tokens import Doc, Span, Token


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    sentence: str


def load_pipeline(model: str = "en_core_web_sm") -> spacy.language.Language:
    return spacy.load(model)


def _span_text(token: Token) -> str:
    # expand a token to its full noun phrase
    span = token.doc[token.left_edge.i: token.right_edge.i + 1]
    return span.text.strip()


def _subtree_verb_text(verb: Token) -> str:
    # include particles or auxiliaries attached to the verb (e.g. 'give up', 'is part of')
    # keep original left-to-right order (for passive voice e.g. 'is awarded')
    parts = [verb]
    for child in verb.children:
        if child.dep_ in ("prt", "aux", "auxpass", "neg"):
            parts.append(child)
    parts.sort(key=lambda x: x.i)
    return " ".join(t.text for t in parts)


def _find_subject(verb: Token) -> Token | None:
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return child
    return None


def _find_object(verb: Token) -> Token | None:
    for child in verb.children:
        if child.dep_ in ("dobj", "attr", "oprd"):
            return child
    return None


def _find_prep_objects(verb: Token) -> Iterator[tuple[str, Token]]:
    for child in verb.children:
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    yield child.text, grandchild


def extract_svo_triples(sent: Span) -> Iterator[Triple]:
    # subject-verb-object and subject-verb-prep-object patterns
    for token in sent:
        if token.pos_ != "VERB" and token.dep_ != "ROOT":
            continue
        if token.pos_ not in ("VERB", "AUX"):
            continue
        
        subj = _find_subject(token)
        if subj is None:
            continue
        
        verb_text = _subtree_verb_text(token)
        subj_text = _span_text(subj)
        
        obj = _find_object(token)
        if obj is not None:
            yield Triple(subj_text)
        
        for prep, pobj in _find_prep_objects(token):
            relation = f"{verb_text} {prep}"
            yield Triple(subj_text, relation, _span_text(pobj), sent.text.strip())


def extract_copula_triples(sent: Span) -> Iterator[Triple]:
    # patterns like 'X is a Y' / 'X is Y' via copula dependency
    for token in sent:
        if token.dep_ != "attr":
            continue
        verb = token.head
        subj = _find_subject(verb)
        if subj is None:
            continue
        yield Triple(_span_text(subj), "is a", _span_text(token), sent.text.strip())


def extract_appos_triples(sent: Span) -> Iterator[Triple]:
    # appositive patterns like 'Marie Curie, a physicist, ...' -> (Marie Curie, is a, physicist)
    for token in sent:
        if token.dep_ == "appos":
            yield Triple(_span_text(token.head), "is a", _span_text(token), sent.text.strip())


def extract_compound_relation(sent: Span) -> Iterator[Triple]:
    # possesive / prep-noun patterns e.g. 'capital of France' -> (France, has, capital)
    for token in sent:
        if token.dep_ == "prep" and token.head.pos_ == "NOUN":
            for child in token.children:
                if child.dep_ == "pobj":
                    yield Triple(_span_text(child), f"has {token.head.text}", _span_text(token.head), sent.text.strip())


EXTRACTORS = (
    extract_svo_triples,
    extract_copula_triples,
    extract_appos_triples,
    extract_compound_relation,
)


def extract_triples(doc: Doc) -> list[Triple]:
    triples: list[Triple] = []
    for sent in doc.sents:
        for extractor in EXTRACTORS:
            triples.extend(extractor(sent))
    return triples
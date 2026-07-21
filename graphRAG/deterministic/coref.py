# heuristic ponoun resolution
# replaces pronouns with the nearest preceding compatible entity mention

import re

from spacy.tokens import Doc, Span

PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their"}

_GENDER_HINTS = {
    "he": "PERSON", "him": "PERSON", "his": "PERSON",
    "she": "PERSON", "her": "PERSON", "hers": "PERSON",
    "it": None, "its": None,
    "they": None, "them": None, "their": None,
}


def _entity_mentions(doc: Doc) -> list[tuple[int, Span]]:
    return [(ent.start, ent) for ent in doc.ents]


def resolve_pronouns(doc: Doc) -> str:
    # return the document text with resolvable pronouns replaced by
    # their most recent matching entity mention
    # sentence-order dependent, single pass
    mentions = _entity_mentions(doc)
    tokens = [t.text_with_ws for t in doc]
    
    last_entity_by_type: dict[str | None, str] = {}
    mention_idx = 0
    
    for i, token in enumerate(doc):
        while mention_idx < len(mentions) and mentions[mention_idx][0] <= i:
            _, ent = mentions[mention_idx]
            last_entity_by_type[ent.label_] = ent.text
            last_entity_by_type[None] = ent.text
            mention_idx += 1
        
        lowered = token.text.lower()
        if lowered in PRONOUNS:
            wanted_type = _GENDER_HINTS.get(lowered)
            replacement = last_entity_by_type.get(wanted_type) or last_entity_by_type.get(None)
            if replacement:
                tokens[i] = replacement + token.whitespace_
    
    return "".join(tokens)


def preprocess_text(nlp, text: str) -> str:
    # run a resolution pass
    doc = nlp(text)
    return resolve_pronouns(doc)
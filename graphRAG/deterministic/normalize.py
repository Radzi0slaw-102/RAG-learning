# deterministic entity normalization
# lowercase/lemma-based canonicalization and alias clustering
# via string similarity

import re
from difflib import SequenceMatcher

_DETERMINERS = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _WHITESPACE.sub(" ", text).strip()
    text = _DETERMINERS.sub("", text)
    return text


def canonical_key(text: str) -> str:
    # lowercased, determiner-stripped key used for exact match and grouping
    return clean_text(text).lower()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class EntityResolver:
    # groups surface forms into canonical entities
    def __init__(self, sim_threshold: float = 0.88):
        self.sim_threshold = sim_threshold
        self._canonical_forms: list[str] = []
        self._alias_to_canonical: dict[str, str] = {}
    
    def resolve (self, surface_form: str) -> str:
        cleaned = clean_text(surface_form)
        key = cleaned.lower()
        
        if key in self._alias_to_canonical:
            return self._alias_to_canonical[key]
        
        for canonical in self._canonical_forms:
            c_key = canonical.lower()
            
            # two forms merge if:
            # - their canonical keys match exactly OR...
            # - one contains the other OR...
            # - they are highly similar strings
            if key == c_key or key in c_key or c_key in key or similarity(key, c_key) >= self.sim_threshold:
                self._alias_to_canonical[key] = canonical
                return canonical
        
        self._canonical_forms.append(cleaned)
        self._alias_to_canonical[key] = cleaned
        return cleaned
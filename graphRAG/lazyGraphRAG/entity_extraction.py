import json
import re
from dataclasses import dataclass, field

from ollama import Client

from config import DEFAULT_MODEL, EXTRACTION_PROMPT


@dataclass
class ExtractedEntity:
    name: str
    type: str
    description: str


@dataclass
class ExtractedRelation:
    source: str
    target: str
    relation: str
    description: str
    source_node_id: str


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


def _parse_json_res(raw: str) -> dict:
    # models could wrap JSON in markdown fences or add stray text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"entities": [], "relations": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}


class EntityExtractor:
    def __init__(self, model: str = DEFAULT_MODEL, host: str | None = 'http://localhost:11434'):
        self.model = model
        self.client = Client(host=host) if host else Client()
    
    def extract(self, text: str, node_id: str) -> ExtractionResult:
        res = self.client.chat(
            model = self.model,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
            format="json",
            options={"temperature": 0.0},
        )
        data = _parse_json_res(res["message"]["content"])
        
        entities = [
            ExtractedEntity(
                name=e["name"].strip(),
                type=e.get("type", "unknown"),
                description=e.get("description", ""),
                source_node_id=node_id,
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]
        relations = [
            ExtractedRelation(
                source=r["source"].strip(),
                target=r["target"].strip(),
                relation=r.get("relation", "related_to"),
                description=r.get("description", ""),
                source_node_id=node_id,
            )
            for r in data.get("relations", [])
            if r.get("source") and r.get("target")
        ]
        return ExtractionResult(entities=entities, relations=relations)
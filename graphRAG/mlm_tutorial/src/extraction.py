import re
from typing import Any

ENTITY_PATTERN = r"entity_name:\ss*(.+?)\s*entity_type:\s*(.+?)\s*entity_description:\s*(.+?)\s*"
RELATIONSHIP_PATTERN = r"source_entity:\s*(.+?)\s*target_entity:\s*(.+?)\s*relation:\s*(.+?)\s*relationship_description:\s*(.+?)\s*"

def parse_fn(response_str: str) -> Any:
    entities = re.findall(ENTITY_PATTERN, response_str)
    relationships = re.findall(RELATIONSHIP_PATTERN, response_str)
    return entities, relationships
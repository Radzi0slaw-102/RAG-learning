DEFAULT_MODEL = "qwen2.5:7b"
EXTRACTION_PROMPT = """You are extracting a knowledge graph from text.

Identify all named entities and the relations between them in the text below.
Entities can be people, organizations, locations, concepts, or other named things.

Return ONLY a JSON object with this exact structure, no other text:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "relation": "...", "description": "..."}}
  ]
}}

Rules:
- Entity names must be normalized (consistent casing, no duplicates).
- Every relation's source/target must match an entity name from the "entities" list.
- Keep descriptions short (one sentence).
- If nothing relevant is found, return empty lists.

Text:
\"\"\"
{text}
\"\"\"
"""
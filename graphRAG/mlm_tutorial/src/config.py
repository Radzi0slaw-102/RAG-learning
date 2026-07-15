OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 20
MAX_PATHS_PER_CHUNK = 2
MAX_CLUSTER_SIZE = 5
REQUEST_TIMEOUT = 300.0
DATA_URL = "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each entity, output exactly one line per field, in this exact format:
entity_name: <name>
entity_type: <type>
entity_description: <description>

2. Identify all pairs of related entities. For each pair, output exactly one line per field, in this exact format:
source_entity: <name>
target_entity: <name>
relation: <relation>
relationship_description: <description>

Do not use any other format. Do not use markdown, numbered lists, or parentheses.

-Real Data-
######################
text: {text}
######################
output:"""
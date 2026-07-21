# How to run
**1. Start Neo4j:**

```sh
docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/cocoindex --name cocoindex-neo4j neo4j:5.26-community
```

**2. Configure .env (example)**

```
COCOINDEX_DB=./cocoindex.db

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=cocoindex
NEO4J_DATABASE=neo4j

LLM_MODEL=ollama/llama3.1:8b
```

**3. Build the graph** - the example ships a `markdown_files/` folder of sample docs so it runs out of the box:

```sh
cocoindex update main
```

To graph your own docs, drop `.md` / `.mdx` files into `markdown_files/` (or point `sourcedir` at your real docs folder) and re-run.

**4. Explore the graph** - open [Neo4j Browser](http://localhost:7474) (`neo4j` / `cocoindex`) and ask:

```cypher
-- How concepts relate
MATCH (a:Entity)-[r:RELATIONSHIP]->(b:Entity)
RETURN a.value, r.predicate, b.value

-- Concepts mentioned in the most documents
MATCH (d:Document)-[:MENTION]->(e:Entity)
RETURN e.value, count(DISTINCT d) AS docs
ORDER BY docs DESC LIMIT 10
```

## Credits
Oryginal repository: https://github.com/cocoindex-io/cocoindex/blob/main/examples/docs_to_knowledge_graph
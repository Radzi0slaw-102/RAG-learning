"""Query pipeline for the CVE/ATT&CK knowledge graph.

Given a question, retrieves a relevant subgraph from Neo4j and asks the
LLM to answer using only that retrieved context.
"""

from __future__ import annotations

import asyncio
import os
import sys

import instructor
import litellm
import pydantic
from neo4j import AsyncGraphDatabase

litellm.drop_params = True

LLM_MODEL = os.environ.get("LLM_MODEL", "ollama/llama3.1:8b")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "cocoindex")

MAX_CVE_NODES_IN_CONTEXT = int(os.environ.get("MAX_CVE_NODES_IN_CONTEXT", "20"))

RETRIEVAL_QUERY = """
MATCH (c:CVE)
WITH c ORDER BY c.cve_id LIMIT $max_cve_nodes
OPTIONAL MATCH (c)-[:HAS_WEAKNESS]->(w:CWE)
OPTIONAL MATCH (c)-[:AFFECTS]->(p:Product)
OPTIONAL MATCH (c)-[:MAPS_TO]->(t:Technique)
OPTIONAL MATCH (t)-[:MITIGATED_BY]->(m:Mitigation)
RETURN c.cve_id AS cve_id, c.description AS description, c.cvss_score AS cvss_score,
       collect(DISTINCT w.cwe_id) AS cwe_ids,
       collect(DISTINCT p.product) AS products,
       collect(DISTINCT {id: t.technique_id, name: t.name, tactic: t.tactic}) AS techniques,
       collect(DISTINCT m.name) AS mitigations
"""

ANSWER_INSTRUCTION = """\
You are a cybersecurity assistant. Answer the question using ONLY the
context provided below. If the context does not contain the answer,
say you don't know. Be concise and specific - give the exact value
(number, ID, name) the question asks for, not a general description.
"""


class Answer(pydantic.BaseModel):
    answer: str = pydantic.Field(description="Concise, specific answer to the question.")


async def fetch_context() -> str:
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        async with driver.session() as session:
            result = await session.run(RETRIEVAL_QUERY, max_cve_nodes=MAX_CVE_NODES_IN_CONTEXT)
            records = [record.data() async for record in result]
    finally:
        await driver.close()

    lines = []
    for r in records:
        techniques = ", ".join(
            f"{t['id']} {t['name']} (tactic: {t['tactic']})"
            for t in r["techniques"] if t.get("id")
        )
        lines.append(
            f"- {r['cve_id']}: {r['description']}\n"
            f"  CVSS score: {r['cvss_score']}\n"
            f"  CWE weaknesses: {', '.join(r['cwe_ids']) or 'none'}\n"
            f"  Affected products: {', '.join(r['products']) or 'none'}\n"
            f"  Maps to ATT&CK techniques: {techniques or 'none'}\n"
            f"  Mitigations: {', '.join(r['mitigations']) or 'none'}"
        )
    return "\n".join(lines)


async def answer_question(question: str, context: str) -> str:
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)
    result = await client.chat.completions.create(
        model=LLM_MODEL,
        response_model=Answer,
        messages=[
            {"role": "system", "content": ANSWER_INSTRUCTION},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return Answer.model_validate(result.model_dump()).answer


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python query_graph.py \"<question>\"")
        return
    question = sys.argv[1]
    context = await fetch_context()
    answer = await answer_question(question, context)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
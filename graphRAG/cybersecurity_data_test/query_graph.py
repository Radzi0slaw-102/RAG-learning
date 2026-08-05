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

# 0 (or unset) means no cap - all CVE nodes are pulled into context
MAX_CVE_NODES_IN_CONTEXT = int(os.environ.get("MAX_CVE_NODES_IN_CONTEXT", "10"))

_LIMIT_CLAUSE = "LIMIT $max_cve_nodes" if MAX_CVE_NODES_IN_CONTEXT > 0 else ""

RETRIEVAL_QUERY = f"""
MATCH (c:CVE)
WITH c ORDER BY c.cve_id {_LIMIT_CLAUSE}
OPTIONAL MATCH (c)-[:HAS_WEAKNESS]->(w:CWE)
OPTIONAL MATCH (c)-[:AFFECTS]->(p:Product)
OPTIONAL MATCH (c)-[r:MAPS_TO]->(t:Technique)
OPTIONAL MATCH (t)-[:MITIGATED_BY]->(m:Mitigation)
RETURN c.cve_id AS cve_id, c.description AS description, c.cvss_score AS cvss_score,
       collect(DISTINCT w.cwe_id) AS cwe_ids,
       collect(DISTINCT p.product) AS products,
       collect(DISTINCT {{id: t.technique_id, name: t.name, tactic: t.tactic, mapping_type: r.mapping_type}}) AS techniques,
       collect(DISTINCT m.name) AS mitigations
"""

ANSWER_INSTRUCTION = """\
You are a cybersecurity assistant. Answer the question using ONLY the
context provided below. If the context does not contain the answer,
say you don't know. Be concise and specific - give the exact value
(number, ID, name) the question asks for, not a general description.

If the context does not contain enough information to answer confidently,
say so plainly in your answer (e.g. "I don't know" or "the context does
not specify this") instead of guessing. Your reasoning must show exactly
which part of the context you used, or state that no relevant part was
found.

You must respond with a JSON object containing exactly these two keys:
- "reasoning": one or two sentences citing the specific fact(s) from the
  context you relied on, or stating that the context lacked the answer.
- "answer": a short, concise string with your final answer.
Do not use any other key names, and do not copy the field names or shape
of the context above.
Example of the exact response shape required:
{"reasoning": "The context lists a CVSS score of 9.8 for this CVE.", "answer": "9.8"}
"""


class Answer(pydantic.BaseModel):
    reasoning: str = pydantic.Field(
        description="Which specific fact(s) from the context were used, or a statement that the context lacked the answer."
    )
    answer: str = pydantic.Field(description="Concise, specific answer to the question.")


async def fetch_context() -> str:
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        params = {"max_cve_nodes": MAX_CVE_NODES_IN_CONTEXT} if MAX_CVE_NODES_IN_CONTEXT > 0 else {}
        async with driver.session() as session:
            result = await session.run(RETRIEVAL_QUERY, **params)
            records = [record.data() async for record in result]
    finally:
        await driver.close()

    lines = []
    for r in records:
        techniques = [t for t in r["techniques"] if t.get("id")]
        exploitation = [t for t in techniques if t.get("mapping_type") == "exploitation_technique"]
        other = [t for t in techniques if t.get("mapping_type") != "exploitation_technique"]
        
        def fmt(t):
            return f"{t['id']} {t['name']} (tactic: {t['tactic']})"
        
        exploitation_str = ", ".join(fmt(t) for t in exploitation) or "none"
        other_str = ", ".join(fmt(t) for t in other) or "none"
        
        lines.append(
            f"- {r['cve_id']}: {r['description']}\n"
            f"  CVSS score: {r['cvss_score']}\n"
            f"  CWE weaknesses: {', '.join(r['cwe_ids']) or 'none'}\n"
            f"  Affected products: {', '.join(r['products']) or 'none'}\n"
            f"  Exploitation technique: {exploitation_str}\n"
            f"  Other associated techniques (impact): {other_str}\n"
            f"  Mitigations: {', '.join(r['mitigations']) or 'none'}"
        )
    return "\n".join(lines)


async def answer_question(question: str, context: str) -> Answer:
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)
    result = await client.chat.completions.create(
        model=LLM_MODEL,
        response_model=Answer,
        messages=[
            {"role": "system", "content": ANSWER_INSTRUCTION},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return Answer.model_validate(result.model_dump())


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
"""Evaluation question generation.

The generated question must not name the source identifier
it is asking about , while still being answerable from that one record.
This forces the downstream RAG pipeline to actually retrieve the right node
from the graph instead of pattern-matching an ID mentioned in the question.
"""

from __future__ import annotations

import asyncio
import json

import instructor
import litellm
import pydantic

litellm.drop_params = True

LLM_MODEL = "ollama/llama3.1:8b"

QUESTION_GEN_INSTRUCTION = """\
You write evaluation questions for a cybersecurity knowledge graph system.

You will be given one fact: an identifier, a field name, and its value.
Write ONE natural-language question that:
- Can only be answered correctly using that fact.
- Does NOT contain the identifier itself, or any other identifier from the
  same family (e.g. do not spell out a CVE number, a CWE number, or a
  technique ID). Refer to the subject descriptively instead (e.g. "the
  Log4j vulnerability disclosed in December 2021" or "the Windows
  privilege escalation flaw known as Zerologon"), using only details
  that are unambiguous and already public knowledge about that entry.
- Is phrased the way a security analyst would actually ask it.

Do not answer the question. Do not explain your reasoning.
"""


class GeneratedQuestion(pydantic.BaseModel):
    question: str = pydantic.Field(description="The natural-language question, with no identifier in it.")


class Fact(pydantic.BaseModel):
    subject_id: str  # the real identifier, kept for grading, never shown to the judge
    subject_hint: str  # a human description usable to build the prompt context
    field: str
    expected_answer: str


async def generate_question(client: instructor.Instructor, fact: Fact) -> str:
    prompt = (
        f"Subject: {fact.subject_hint}\n"
        f"Field being asked about: {fact.field}\n"
        f"Known value (for your context only, do not reveal verbatim if it's the identifier itself): {fact.expected_answer}\n"
    )
    result = await client.chat.completions.create(
        model=LLM_MODEL,
        response_model=GeneratedQuestion,
        messages=[
            {"role": "system", "content": QUESTION_GEN_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
    )
    return GeneratedQuestion.model_validate(result.model_dump()).question


def build_facts() -> list[Fact]:
    cve_data = json.loads(open("cve_records.json").read())["cve_records"]
    attack_data = json.loads(open("attack_techniques.json").read())["techniques"]
    mapping_data = json.loads(open("kev_attack_mapping.json").read())["mapping_objects"]

    facts: list[Fact] = []

    for r in cve_data:
        hint = f"the vulnerability described as: \"{r['description'][:160]}...\""
        facts.append(Fact(
            subject_id=r["cve_id"],
            subject_hint=hint,
            field="CVSS score",
            expected_answer=str(r["cvss_score"]),
        ))
        facts.append(Fact(
            subject_id=r["cve_id"],
            subject_hint=hint,
            field="associated CWE weakness type(s)",
            expected_answer=", ".join(r["cwe_ids"]),
        ))
        facts.append(Fact(
            subject_id=r["cve_id"],
            subject_hint=hint,
            field="affected vendor/product",
            expected_answer=", ".join(f"{p['vendor']} {p['product']}" for p in r["affected_products"]),
        ))

    for t in attack_data:
        hint = f"the ATT&CK entry described as: \"{t['description'][:160]}...\""
        facts.append(Fact(
            subject_id=t["technique_id"],
            subject_hint=hint,
            field="ATT&CK tactic category",
            expected_answer=t["tactic"],
        ))
        facts.append(Fact(
            subject_id=t["technique_id"],
            subject_hint=hint,
            field="recommended mitigations",
            expected_answer=", ".join(t["mitigations"]),
        ))

    seen_pairs: set[tuple[str, str]] = set()
    for m in mapping_data:
        pair = (m["capability_id"], m["attack_object_id"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        cve_desc = next((r["description"][:120] for r in cve_data if r["cve_id"] == m["capability_id"]), m["capability_description"])
        facts.append(Fact(
            subject_id=f"{m['capability_id']}->{m['attack_object_id']}",
            subject_hint=f"the vulnerability described as: \"{cve_desc}...\"",
            field="ATT&CK technique it maps to",
            expected_answer=f"{m['attack_object_id']} ({m['attack_object_name']})",
        ))

    return facts


async def main() -> None:
    facts = build_facts()
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)

    entries = []
    for fact in facts:
        question = await generate_question(client, fact)
        entries.append({
            "subject_id": fact.subject_id,
            "field": fact.field,
            "question": question,
            "expected_answer": fact.expected_answer,
        })
        print(f"[{fact.subject_id}] {fact.field} -> {question}")

    with open("eval_questions.json", "w") as f:
        json.dump({"questions": entries}, f, indent=2)
    print(f"\nWrote {len(entries)} questions to eval_questions.json")


if __name__ == "__main__":
    asyncio.run(main())
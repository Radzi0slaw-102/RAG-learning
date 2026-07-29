"""Evaluation question generation.

The generated question must not name the source identifier
it is asking about, while still being answerable from that one record.
This forces the downstream RAG pipeline to actually retrieve the right node
from the graph instead of pattern-matching an ID mentioned in the question.
"""

from __future__ import annotations

import asyncio
import json
import random

import instructor
import litellm
import pydantic

litellm.drop_params = True

LLM_MODEL = "ollama/llama3.1:8b"
RANDOM_SEED = 42

QUESTION_GEN_INSTRUCTION = """\
You write complex evaluation questions designed to stress-test a GraphRAG system built on a cybersecurity knowledge graph.
 
You will be given a target entity (e.g., a vulnerability, APT group, technique, or product) and a specific target fact connected to it.
 
For each generated entry:
1. Provide your reasoning - Explain the logical chain and graph relationships connecting the target entity to the answer without explicitly naming primary identifiers.
2. Write ONE natural-language question that:
   - requires multi-hop graph traversal: The question must force the system to follow non-obvious, indirect relationships in the graph (e.g., linking a vulnerability through its associated CWE/technique to a threat actor, campaign, or targeted sector).
   - avoids direct identifiers: Do not include primary identifiers (such as CVE IDs, CWE IDs, TTP codes, or APT numbers). Describe entities functionally or contextually (e.g., "the remote code execution flaw in Apache Log4j disclosed in late 2021" or "the threat group known for targeting energy sectors using living-off-the-land techniques").
   - reflects real-world cybersecurity intent: Phrase the question like a curious threat hunter, incident responder, or SOC analyst investigating broader impacts, contextual risks, or hidden overlaps rather than looking up a static database entry.
 
Do not answer the question.
"""

YES_NO_QUESTION_GEN_INSTRUCTION = """\
You write complex yes/no evaluation questions designed to stress-test a GraphRAG system built on a cybersecurity knowledge graph.
 
You will be given a technique description and a candidate mitigation name.
 
For each generated entry:
1. Provide your reasoning - Explain the logical chain connecting the technique's behavior to whether the candidate mitigation would plausibly defend against it, without explicitly naming primary identifiers.
2. Write ONE yes/no question that:
   - avoids direct identifiers: Describe the technique functionally (what it does), never by its ID.
   - names the candidate mitigation explicitly in the question.
   - reflects real-world cybersecurity intent: Phrase it the way a security analyst assessing their defenses would ask it.
 
Do not answer the question.
"""


class GeneratedQuestion(pydantic.BaseModel):
    reasoning: str = pydantic.Field(description="The graph-relationship reasoning connecting the entity to the answer. Shown to human reviewers only, never to the system being evaluated.")
    question: str = pydantic.Field(description="The natural-language question, with no identifier in it.")


class Fact(pydantic.BaseModel):
    subject_id: str  # the real identifier, kept for grading, never shown to the judge
    subject_hint: str  # a human description usable to build the prompt context
    field: str
    expected_answer: str


class MitigationCheckFact(pydantic.BaseModel):
    subject_id: str
    technique_hint: str
    candidate_mitigation: str
    expected_answer: str  # "Yes" or "No"


def _is_usable_answer(value: str) -> bool:
    if not value.strip():
        return False
    tokens = value.replace(",", " ").split()
    return any(t.strip().lower() not in ("n/a", "n\\a", "none", "unknown") for t in tokens)


async def generate_question(client: instructor.Instructor, fact: Fact) -> GeneratedQuestion:
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
    return GeneratedQuestion.model_validate(result.model_dump())


async def generate_mitigation_question(client: instructor.Instructor, fact: MitigationCheckFact) -> GeneratedQuestion:
    prompt = (
        f"Technique: {fact.technique_hint}\n"
        f"Candidate mitigation: {fact.candidate_mitigation}\n"
    )
    result = await client.chat.completions.create(
        model=LLM_MODEL,
        response_model=GeneratedQuestion,
        messages=[
            {"role": "system", "content": YES_NO_QUESTION_GEN_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
    )
    return GeneratedQuestion.model_validate(result.model_dump())


def build_facts() -> tuple[list[Fact], list[MitigationCheckFact]]:
    cve_data = json.loads(open("data/cve_records.json").read())["cve_records"]
    attack_data = json.loads(open("data/attack_techniques.json").read())["techniques"]
    mapping_data = json.loads(open("data/kev_attack_mapping.json").read())["mapping_objects"]
    
    facts: list[Fact] = []
    skipped: list[str] = []
    
    for r in cve_data:
        hint = f"the vulnerability described as: \"{r['description'][:160]}...\""

        cvss_answer = str(r["cvss_score"])
        if _is_usable_answer(cvss_answer):
            facts.append(Fact(
                subject_id=r["cve_id"],
                subject_hint=hint,
                field="CVSS score",
                expected_answer=cvss_answer,
            ))
        else:
            skipped.append(f"{r['cve_id']} / CVSS score")

        cwe_answer = ", ".join(r["cwe_ids"])
        if _is_usable_answer(cwe_answer):
            facts.append(Fact(
                subject_id=r["cve_id"],
                subject_hint=hint,
                field="associated CWE weakness type(s)",
                expected_answer=cwe_answer,
            ))
        else:
            skipped.append(f"{r['cve_id']} / associated CWE weakness type(s)")

        product_answer = ", ".join(f"{p['vendor']} {p['product']}" for p in r["affected_products"])
        if _is_usable_answer(product_answer):
            facts.append(Fact(
                subject_id=r["cve_id"],
                subject_hint=hint,
                field="affected vendor/product",
                expected_answer=product_answer,
            ))
        else:
            skipped.append(f"{r['cve_id']} / affected vendor/product")
    
    for t in attack_data:
        hint = f"the ATT&CK entry described as: \"{t['description'][:160]}...\""
        tactic_answer = t["tactic"]
        if _is_usable_answer(tactic_answer):
            facts.append(Fact(
                subject_id=t["technique_id"],
                subject_hint=hint,
                field="ATT&CK tactic category",
                expected_answer=tactic_answer,
            ))
        else:
            skipped.append(f"{t['technique_id']} / ATT&CK tactic category")
    
    exploitation_mappings = [m for m in mapping_data if m["mapping_type"] == "exploitation_technique"]
    seen_cves: set[str] = set()
    for m in exploitation_mappings:
        if m["capability_id"] in seen_cves:
            continue
        seen_cves.add(m["capability_id"])
        cve_desc = next((r["description"][:120] for r in cve_data if r["cve_id"] == m["capability_id"]), m["capability_description"])
        mapping_answer = f"{m['attack_object_id']} ({m['attack_object_name']})"
        if _is_usable_answer(mapping_answer):
            facts.append(Fact(
                subject_id=f"{m['capability_id']}->{m['attack_object_id']}",
                subject_hint=f"the vulnerability described as: \"{cve_desc}...\"",
                field="ATT&CK exploitation technique",
                expected_answer=mapping_answer,
            ))
        else:
            skipped.append(f"{m['capability_id']}->{m['attack_object_id']} / ATT&CK exploitation technique")
    
    rng = random.Random(RANDOM_SEED)
    all_mitigations = sorted({m for t in attack_data for m in t["mitigations"]})
    
    mitigation_facts: list[MitigationCheckFact] = []
    for t in attack_data:
        hint = f"the ATT&CK entry described as: \"{t['description'][:160]}...\""
        true_mitigations = t["mitigations"]
        false_candidates = [m for m in all_mitigations if m not in true_mitigations]
 
        for m in rng.sample(true_mitigations, k=min(2, len(true_mitigations))):
            mitigation_facts.append(MitigationCheckFact(
                subject_id=t["technique_id"],
                technique_hint=hint,
                candidate_mitigation=m,
                expected_answer="Yes",
            ))
        for m in rng.sample(false_candidates, k=min(2, len(false_candidates))):
            mitigation_facts.append(MitigationCheckFact(
                subject_id=t["technique_id"],
                technique_hint=hint,
                candidate_mitigation=m,
                expected_answer="No",
            ))

    if skipped:
        print(f"Skipped {len(skipped)} fact(s) with no usable ground truth in the source data:")
        for s in skipped:
            print(f"  - {s}")
 
    return facts, mitigation_facts


async def main() -> None:
    facts, mitigation_facts = build_facts()
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)
    
    entries = []
    for fact in facts:
        generated = await generate_question(client, fact)
        entries.append({
            "subject_id": fact.subject_id,
            "field": fact.field,
            "reasoning": generated.reasoning,
            "question": generated.question,
            "expected_answer": fact.expected_answer,
        })
        print(f"[{fact.subject_id}] {fact.field} -> {generated.question} -> {fact.expected_answer}")
    
    for fact in mitigation_facts:
        generated = await generate_mitigation_question(client, fact)
        entries.append({
            "subject_id": fact.subject_id,
            "field": "mitigation check (yes/no)",
            "reasoning": generated.reasoning,
            "question": generated.question,
            "expected_answer": fact.expected_answer,
        })
        print(f"[{fact.subject_id}] mitigation check ({fact.candidate_mitigation}) -> {generated.question} -> {fact.expected_answer}")
    
    with open("results/eval_questions.json", "w") as f:
        json.dump({"questions": entries}, f, indent=2)
    print(f"\nWrote {len(entries)} questions to eval_questions.json")


if __name__ == "__main__":
    asyncio.run(main())
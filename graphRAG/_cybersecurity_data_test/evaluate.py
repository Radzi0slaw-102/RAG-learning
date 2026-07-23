"""Evaluation loop.

For each question in eval_questions.json: retrieve context from Neo4j,
get the RAG pipeline's answer, then ask an LLM judge whether that answer
matches the known ground truth. Prints a per-question verdict and a
final accuracy summary.
"""

from __future__ import annotations

import asyncio
import json

import instructor
import litellm
import pydantic

from query_graph import answer_question, fetch_context

litellm.drop_params = True

LLM_MODEL = "ollama/llama3.1:8b"

JUDGE_INSTRUCTION = """\
You are grading a cybersecurity Q&A system. You will see a question, the
system's answer, and the known correct answer. Decide if the system's
answer is correct: it must convey the same specific fact as the correct
answer (matching IDs, numbers, or names), not just be topically related.
Minor wording differences are fine. Missing or vague answers are incorrect.
"""


class Verdict(pydantic.BaseModel):
    correct: bool = pydantic.Field(description="True if the system answer matches the correct answer.")
    reasoning: str = pydantic.Field(description="One short sentence explaining the verdict.")


async def judge(client: instructor.Instructor, question: str, system_answer: str, expected: str) -> Verdict:
    prompt = (
        f"Question: {question}\n"
        f"System answer: {system_answer}\n"
        f"Correct answer: {expected}\n"
    )
    result = await client.chat.completions.create(
        model=LLM_MODEL,
        response_model=Verdict,
        messages=[
            {"role": "system", "content": JUDGE_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
    )
    return Verdict.model_validate(result.model_dump())


async def main() -> None:
    data = json.loads(open("eval_questions.json").read())
    questions = data["questions"]

    context = await fetch_context()
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)

    results = []
    for entry in questions:
        system_answer = await answer_question(entry["question"], context)
        verdict = await judge(client, entry["question"], system_answer, entry["expected_answer"])

        results.append({
            "subject_id": entry["subject_id"],
            "field": entry["field"],
            "question": entry["question"],
            "expected_answer": entry["expected_answer"],
            "system_answer": system_answer,
            "correct": verdict.correct,
            "judge_reasoning": verdict.reasoning,
        })

        mark = "PASS" if verdict.correct else "FAIL"
        print(f"[{mark}] {entry['subject_id']} / {entry['field']}")
        print(f"  Q: {entry['question']}")
        print(f"  Got: {system_answer}")
        print(f"  Expected: {entry['expected_answer']}")
        print(f"  Judge: {verdict.reasoning}\n")

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0.0

    with open("eval_results.json", "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)

    print(f"Accuracy: {accuracy:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")
    print("Full results written to eval_results.json")


if __name__ == "__main__":
    asyncio.run(main())
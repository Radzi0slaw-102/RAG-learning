"""Evaluation loop.

For each question in eval_questions.json: retrieve context from Neo4j,
get the RAG pipeline's answer, then ask an LLM judge whether that answer
matches the known ground truth. Prints a per-question verdict and a
final accuracy summary.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict

import instructor
import litellm
import pydantic

from query_graph import answer_question, fetch_context

litellm.drop_params = True

LLM_MODEL = os.environ.get("LLM_MODEL", "ollama/llama3.1:8b")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

JUDGE_INSTRUCTION = """\
You are grading a cybersecurity Q&A system. You will see a question, the
system's answer, and the known correct answer. Decide if the system's
answer is correct: it must convey the same specific fact as the correct
answer (matching IDs, numbers, or names), not just be topically related.
Minor wording differences are fine. Missing or vague answers are incorrect.
"""

YES_NO_JUDGE_INSTRUCTION = """\
You are grading a cybersecurity Q&A system on a yes/no question. You
will see the question, the system's answer, and the correct answer
("Yes" or "No"). Case does not matter ("yes"/"Yes"/"YES" are identical).
Treat any clear affirmative ("yes", "it is", "it would", "correct") as
agreeing with "Yes", and any clear negative ("no", "it is not", "it
would not", "unrelated") as agreeing with "No" — regardless of extra
explanation attached, as long as the first clear yes/no signal in the
answer is unambiguous. If the system's answer contradicts itself or
gives no clear yes/no signal, mark it incorrect.
"""

UNANSWERABLE_JUDGE_INSTRUCTION = """\
You are grading a cybersecurity Q&A system on a question that has NO
answer in the system's knowledge base (the subject does not exist in
the data). The correct behavior is for the system to say it doesn't
know or that it has no information on the subject. Mark it correct
if the answer plainly admits it lacks the information (e.g. "I don't
know", "the context does not specify this", "no information found",
"there's no record of this"). Mark it incorrect if the answer states
any specific fact, number, ID, or name as if it were real, even
hedged - fabricating a plausible-sounding value is a failure here,
not a partial success.
"""


class Verdict(pydantic.BaseModel):
    correct: bool = pydantic.Field(description="True if the system answer matches the correct answer.")
    reasoning: str = pydantic.Field(description="One short sentence explaining the verdict.")


def _normalize_yes_no(text: str) -> str | None:
    stripped = text.strip().lower()
    if stripped.startswith("yes"):
        return "yes"
    if stripped.startswith("no"):
        return "no"
    return None


async def judge(client: instructor.Instructor, question: str, system_answer: str, expected: str, field: str) -> Verdict:
    if field == "mitigation check (yes/no)":
        normalized_system = _normalize_yes_no(system_answer)
        normalized_expected = _normalize_yes_no(expected)
        # unambiguous leading Yes/No on both sides, skip the LLM judge entirely and compare directly
        if normalized_system is not None and normalized_expected is not None:
            correct = normalized_system == normalized_expected
            return Verdict(
                correct=correct,
                reasoning=f"Direct comparison: system said '{normalized_system}', expected '{normalized_expected}'.",
            )
        instruction = YES_NO_JUDGE_INSTRUCTION
    elif field == "unanswerable":
        instruction = UNANSWERABLE_JUDGE_INSTRUCTION
    else:
        instruction = JUDGE_INSTRUCTION
    
    prompt = (
        f"Question: {question}\n"
        f"System answer: {system_answer}\n"
        f"Correct answer: {expected}\n"
    )
    result = await client.chat.completions.create(
        model=LLM_MODEL,
        response_model=Verdict,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
    )
    return Verdict.model_validate(result.model_dump())


async def main() -> None:
    data = json.loads(open(f"{RESULTS_DIR}/eval_questions.json").read())
    questions = data["questions"]

    context = await fetch_context()
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)

    results = []
    n_errors = 0
    for entry in questions:
        try:
            system_result = await answer_question(entry["question"], context)
            system_answer = system_result.answer
            system_reasoning = system_result.reasoning
        except Exception as exc:
            n_errors += 1
            results.append({
                "subject_id": entry["subject_id"],
                "field": entry["field"],
                "question_reasoning": entry.get("reasoning", ""),
                "question": entry["question"],
                "expected_answer": entry["expected_answer"],
                "system_answer": f"[ERROR: {type(exc).__name__}]",
                "system_reasoning": "",
                "correct": False,
                "judge_reasoning": "Skipped: the system failed to produce a valid answer for this question.",
            })
            print(f"[ERROR] {entry['subject_id']} / {entry['field']} -> {type(exc).__name__}: {exc}\n")
            continue

        try:
            verdict = await judge(client, entry["question"], system_answer, entry["expected_answer"], entry["field"])
        except Exception as exc:
            n_errors += 1
            results.append({
                "subject_id": entry["subject_id"],
                "field": entry["field"],
                "question_reasoning": entry.get("reasoning", ""),
                "question": entry["question"],
                "expected_answer": entry["expected_answer"],
                "system_answer": system_answer,
                "system_reasoning": system_reasoning,
                "correct": False,
                "judge_reasoning": f"Skipped: the judge failed ({type(exc).__name__}).",
            })
            print(f"[JUDGE ERROR] {entry['subject_id']} / {entry['field']} -> {type(exc).__name__}: {exc}\n")
            continue

        results.append({
            "subject_id": entry["subject_id"],
            "field": entry["field"],
            "question_reasoning": entry.get("reasoning", ""),
            "question": entry["question"],
            "expected_answer": entry["expected_answer"],
            "system_answer": system_answer,
            "system_reasoning": system_reasoning,
            "correct": verdict.correct,
            "judge_reasoning": verdict.reasoning,
        })

        mark = "PASS" if verdict.correct else "FAIL"
        print(f"[{mark}] {entry['subject_id']} / {entry['field']}")
        print(f"  Q: {entry['question']}")
        print(f"  Got: {system_answer}")
        print(f"  System reasoning: {system_reasoning}")
        print(f"  Expected: {entry['expected_answer']}")
        print(f"  Judge: {verdict.reasoning}\n")

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0.0

    by_field = defaultdict(lambda: [0, 0])
    for r in results:
        by_field[r["field"]][1] += 1
        if r["correct"]:
            by_field[r["field"]][0] += 1

    with open(f"{RESULTS_DIR}/eval_results.json", "w") as f:
        json.dump({"accuracy": accuracy, "errors": n_errors, "by_field": {k: f"{c}/{t}" for k, (c, t) in by_field.items()}, "results": results}, f, indent=2)

    print(f"Accuracy: {accuracy:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")
    if n_errors:
        print(f"Errors (excluded from correct, counted in total): {n_errors}")
    print("By field:")
    for field, (correct, total) in sorted(by_field.items()):
        print(f"  {field}: {correct}/{total} ({correct/total:.1%})")
    print("Full results written to eval_results.json")


if __name__ == "__main__":
    asyncio.run(main())
"""
Score each model's results/raw/{model}.jsonl with ragas and write a
per-model summary to results/scores.csv.

Note: ragas==0.4.x's evaluate() rejects the new ragas.metrics.collections
metric instances due to a stale isinstance(m, Metric) check (see
https://github.com/explodinggradients/ragas/issues/2624), even though
the collections API is the one the library itself recommends. We call
metric.ascore() directly per sample instead of going through evaluate().
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from rag_pipeline import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "raw"
SCORES_PATH = PROJECT_ROOT / "results" / "scores.csv"


def build_judge(config: dict):
    client = AsyncOpenAI(api_key="ollama", base_url=f"{config['ollama_base_url']}/v1")
    llm = llm_factory(model=config["judge_model"], provider="openai", client=client, max_tokens=4096)
    embeddings = OpenAIEmbeddings(client=client, model=config["embedding_model"])
    return llm, embeddings


def load_records(model_name: str) -> list[dict]:
    path = RESULTS_DIR / f"{model_name.replace(':', '_')}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


async def score_model(model_name: str, judge_llm, judge_embeddings) -> dict:
    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
    context_precision = ContextPrecision(llm=judge_llm)
    context_recall = ContextRecall(llm=judge_llm)

    records = load_records(model_name)
    totals = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}

    for record in records:
        faith = await faithfulness.ascore(
            user_input=record["question"], response=record["answer"], retrieved_contexts=record["contexts"]
        )
        relevancy = await answer_relevancy.ascore(
            user_input=record["question"], response=record["answer"]
        )
        precision = await context_precision.ascore(
            user_input=record["question"],
            retrieved_contexts=record["contexts"],
            reference=record["ground_truth"],
        )
        recall = await context_recall.ascore(
            user_input=record["question"],
            retrieved_contexts=record["contexts"],
            reference=record["ground_truth"],
        )
        totals["faithfulness"] += faith.value
        totals["answer_relevancy"] += relevancy.value
        totals["context_precision"] += precision.value
        totals["context_recall"] += recall.value

    n = len(records)
    return {"model": model_name, **{k: v / n for k, v in totals.items()}}


async def main() -> None:
    config = load_config()
    judge_llm, judge_embeddings = build_judge(config)

    rows = []
    for model in config["llm_models"]:
        row = await score_model(model["name"], judge_llm, judge_embeddings)
        rows.append(row)
        print(row)

    pd.DataFrame(rows).to_csv(SCORES_PATH, index=False)
    print(f"Wrote {SCORES_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
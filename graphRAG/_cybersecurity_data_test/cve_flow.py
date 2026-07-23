# CVE ingestion flow
from __future__ import annotations

import asyncio
import json
import os
import pathlib
from collections.abc import AsyncIterator
from dataclasses import dataclass

import instructor
import litellm

import cocoindex as coco
from cocoindex.connectors import neo4j
from cocoindex.resources.id import generate_id

from schemas import (
    CVE_EXTRACTION_INSTRUCTION,
    Affects,
    CveNode,
    CweNode,
    ExtractedVulnerabilityType,
    HasWeakness,
    ProductNode,
)

litellm.drop_params = True

KG_DB = coco.ContextKey[neo4j.ConnectionFactory]("kg_db")
LLM_MODEL = coco.ContextKey[str]("llm_model", detect_change=True)


@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    builder.provide(
        KG_DB,
        neo4j.ConnectionFactory(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.environ.get("NEO4J_USER", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", "cocoindex"),
            ),
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        ),
    )
    builder.provide(LLM_MODEL, os.environ.get("LLM_MODEL", "ollama/llama3.1:8b"))
    yield


@dataclass
class RawCveRecord:
    cve_id: str
    description: str
    cvss_score: float
    published_date: str
    cwe_ids: list[str]
    affected_products: list[dict]


@coco.fn(memo=True)
async def extract_vulnerability_type(description: str) -> ExtractedVulnerabilityType:
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)
    result = await client.chat.completions.create(
        model=coco.use_context(LLM_MODEL),
        response_model=ExtractedVulnerabilityType,
        messages=[
            {"role": "system", "content": CVE_EXTRACTION_INSTRUCTION},
            {"role": "user", "content": description},
        ],
    )
    return ExtractedVulnerabilityType.model_validate(result.model_dump())


@coco.fn(memo=True)
async def process_cve(
    record: RawCveRecord,
    cve_table: neo4j.TableTarget[CveNode],
    cwe_table: neo4j.TableTarget[CweNode],
    product_table: neo4j.TableTarget[ProductNode],
    has_weakness_rel: neo4j.RelationTarget[HasWeakness],
    affects_rel: neo4j.RelationTarget[Affects],
) -> ExtractedVulnerabilityType:
    extracted = await extract_vulnerability_type(record.description)
    
    cve_table.declare_record(
        row=CveNode(
            cve_id=record.cve_id,
            description=record.description,
            cvss_score=record.cvss_score,
            published_date=record.published_date,
        )
    )
    
    for cwe_id in record.cwe_ids:
        cwe_table.declare_record(row=CweNode(cwe_id=cwe_id, name=cwe_id))
        edge_id = await generate_id((record.cve_id, cwe_id))
        has_weakness_rel.declare_relation(
            from_id=record.cve_id,
            to_id=cwe_id,
            record=HasWeakness(id=edge_id),
        )
    
    for product in record.affected_products:
        key = f"{product['vendor']}::{product['product']}"
        product_table.declare_record(
            row=ProductNode(key=key, vendor=product["vendor"], product=product["product"])
        )
        edge_id = await generate_id((record.cve_id, key))
        affects_rel.declare_relation(
            from_id=record.cve_id,
            to_id=key,
            record=Affects(id=edge_id),
        )
    
    return extracted


@coco.fn
async def app_main(source: pathlib.Path) -> None:
    cve_table = await neo4j.mount_table_target(
        KG_DB,
        "CVE",
        await neo4j.TableSchema.from_class(CveNode, primary_key="cve_id"),
        primary_key="cve_id",
    )
    cwe_table = await neo4j.mount_table_target(
        KG_DB,
        "CWE",
        await neo4j.TableSchema.from_class(CweNode, primary_key="cwe_id"),
        primary_key="cwe_id",
    )
    product_table = await neo4j.mount_table_target(
        KG_DB,
        "Product",
        await neo4j.TableSchema.from_class(ProductNode, primary_key="key"),
        primary_key="key",
    )
    has_weakness_rel = await neo4j.mount_relation_target(
        KG_DB,
        "HAS_WEAKNESS",
        cve_table,
        cwe_table,
        await neo4j.TableSchema.from_class(HasWeakness, primary_key="id"),
        primary_key="id",
    )
    affects_rel = await neo4j.mount_relation_target(
        KG_DB,
        "AFFECTS",
        cve_table,
        product_table,
        await neo4j.TableSchema.from_class(Affects, primary_key="id"),
        primary_key="id",
    )
    
    raw = json.loads(source.read_text())
    records = [
        RawCveRecord(
            cve_id=r["cve_id"],
            description=r["description"],
            cvss_score=r["cvss_score"],
            published_date=r["published_date"],
            cwe_ids=r["cwe_ids"],
            affected_products=r["affected_products"],
        )
        for r in raw["cve_records"]
    ]
    
    coros = [
        coco.use_mount(
            coco.component_subpath("cve", rec.cve_id),
            process_cve,
            rec,
            cve_table,
            cwe_table,
            product_table,
            has_weakness_rel,
            affects_rel,
        )
        for rec in records
    ]
    results = await asyncio.gather(*coros)
    for rec, extracted in zip(records, results):
        print(f"{rec.cve_id}: {extracted.vulnerability_type} — {extracted.attack_vector_summary}")


_source_path = pathlib.Path(os.environ.get("SOURCE_PATH", "data/raw/cve_records.json"))

app = coco.App(
    coco.AppConfig(name="CveIngestion"),
    app_main,
    source=_source_path,
)
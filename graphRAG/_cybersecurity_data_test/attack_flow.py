# MITRE ATT&CK ingestion flow
from __future__ import annotations

import asyncio
import json
import os
import pathlib
from collections.abc import AsyncIterator
from dataclasses import dataclass

import cocoindex as coco
from cocoindex.connectors import neo4j
from cocoindex.resources.id import generate_id

from schemas import MitigatedBy, MitigationNode, TechniqueNode

KG_DB = coco.ContextKey[neo4j.ConnectionFactory]("kg_db")


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
    yield


@dataclass
class RawTechnique:
    technique_id: str
    name: str
    tactic: str
    description: str
    mitigations: list[str]


@coco.fn(memo=True)
async def process_technique(
    record: RawTechnique,
    technique_table: neo4j.TableTarget[TechniqueNode],
    mitigation_table: neo4j.TableTarget[MitigationNode],
    mitigated_by_rel: neo4j.RelationTarget[MitigatedBy],
) -> None:
    technique_table.declare_record(
        row=TechniqueNode(
            technique_id=record.technique_id,
            name=record.name,
            tactic=record.tactic,
            description=record.description,
        )
    )

    for mitigation_name in record.mitigations:
        mitigation_table.declare_record(row=MitigationNode(name=mitigation_name))
        edge_id = await generate_id((record.technique_id, mitigation_name))
        mitigated_by_rel.declare_relation(
            from_id=record.technique_id,
            to_id=mitigation_name,
            record=MitigatedBy(id=edge_id),
        )


@coco.fn
async def app_main(source: pathlib.Path) -> None:
    technique_table = await neo4j.mount_table_target(
        KG_DB,
        "Technique",
        await neo4j.TableSchema.from_class(TechniqueNode, primary_key="technique_id"),
        primary_key="technique_id",
    )
    mitigation_table = await neo4j.mount_table_target(
        KG_DB,
        "Mitigation",
        await neo4j.TableSchema.from_class(MitigationNode, primary_key="name"),
        primary_key="name",
    )
    mitigated_by_rel = await neo4j.mount_relation_target(
        KG_DB,
        "MITIGATED_BY",
        technique_table,
        mitigation_table,
        await neo4j.TableSchema.from_class(MitigatedBy, primary_key="id"),
        primary_key="id",
    )

    raw = json.loads(source.read_text())
    records = [
        RawTechnique(
            technique_id=t["technique_id"],
            name=t["name"],
            tactic=t["tactic"],
            description=t["description"],
            mitigations=t["mitigations"],
        )
        for t in raw["techniques"]
    ]

    coros = [
        coco.use_mount(
            coco.component_subpath("technique", rec.technique_id),
            process_technique,
            rec,
            technique_table,
            mitigation_table,
            mitigated_by_rel,
        )
        for rec in records
    ]
    await asyncio.gather(*coros)
    for rec in records:
        print(f"{rec.technique_id}: {rec.name} ({rec.tactic})")


_source_path = pathlib.Path(os.environ.get("SOURCE_PATH", "data/attack_techniques.json"))

app = coco.App(
    coco.AppConfig(name="AttackIngestion"),
    app_main,
    source=_source_path,
)
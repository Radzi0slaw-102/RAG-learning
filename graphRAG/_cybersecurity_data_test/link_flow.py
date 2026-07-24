# CVE-to-ATT&CK linking flow.

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

from schemas import CveNode, TechniqueNode

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
class MapsTo:
    id: str


@dataclass
class RawMapping:
    cve_id: str
    technique_id: str
    mapping_type: str


@coco.fn(memo=True)
async def process_mapping(
    record: RawMapping,
    maps_to_rel: neo4j.RelationTarget[MapsTo],
) -> None:
    edge_id = await generate_id((record.cve_id, record.technique_id, record.mapping_type))
    maps_to_rel.declare_relation(
        from_id=record.cve_id,
        to_id=record.technique_id,
        record=MapsTo(id=edge_id),
    )


@coco.fn
async def app_main(source: pathlib.Path) -> None:
    cve_table = neo4j.declare_table_target(
        KG_DB,
        "CVE",
        await neo4j.TableSchema.from_class(CveNode, primary_key="cve_id"),
        primary_key="cve_id",
    )
    technique_table = neo4j.declare_table_target(
        KG_DB,
        "Technique",
        await neo4j.TableSchema.from_class(TechniqueNode, primary_key="technique_id"),
        primary_key="technique_id",
    )
    maps_to_rel = await neo4j.mount_relation_target(
        KG_DB,
        "MAPS_TO",
        cve_table,
        technique_table,
        await neo4j.TableSchema.from_class(MapsTo, primary_key="id"),
        primary_key="id",
    )

    raw = json.loads(source.read_text())
    records = [
        RawMapping(
            cve_id=m["capability_id"],
            technique_id=m["attack_object_id"],
            mapping_type=m["mapping_type"],
        )
        for m in raw["mapping_objects"]
    ]

    coros = [
        coco.use_mount(
            coco.component_subpath("mapping", f"{rec.cve_id}-{rec.technique_id}-{rec.mapping_type}"),
            process_mapping,
            rec,
            maps_to_rel,
        )
        for rec in records
    ]
    await asyncio.gather(*coros)
    print(f"Declared {len(records)} MAPS_TO edges.")


_source_path = pathlib.Path(os.environ.get("SOURCE_PATH", "data/kev_attack_mapping.json"))

app = coco.App(
    coco.AppConfig(name="CveAttackLinking"),
    app_main,
    source=_source_path,
)
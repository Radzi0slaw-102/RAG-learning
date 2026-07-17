from __future__ import annotations

import asyncio
import os
import pathlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import instructor
import litellm
import pydantic

import cocoindex as coco
from cocoindex.connectors import localfs, neo4j
from cocoindex.resources.file import PatternFilePathMatcher
from cocoindex.resources.id import generate_id

litellm.drop_params = True

KG_DB = coco.ContextKey[neo4j.ConnectionFactory]("kg_db")
LLM_MODEL = coco.ContextKey[str]("llm_model", detect_change=True)

# runtime lifecycle
# - setup runs when the runtime starts
# - cleanup runs when the runtime stops
# used to configure settings programmatically
# for simplier uses replace function call with COCOINDEX_DB env variable
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

# neo4j node and edge schemas

@dataclass
class Document:
    filename: str # primary key
    title: str
    summary: str


@dataclass
class Entity:
    value: str


@dataclass
class Relationship:
    """id is a stable hash of the triple so
    the same (subject, predicate, object) always maps to a single edge
    The predicate is stored as an edge property."""
    id: int
    predicate: str


# MENTION is declared without a schema


# LLM extraction schemas (pydantic for instructor):

class DocumentSummary(pydantic.BaseModel):
    title: str = pydantic.Field(description="A concise title for document.")
    summary: str = pydantic.Field(
        description="A one-paragraph summary of what the document covers."
    )


class ExtractedRelationship(pydantic.BaseModel):
    subject: str = pydantic.Field(
        description="The concept the statement is about, e.g. 'CocoIndex'."
    )
    predicate: str = pydantic.Field(
        description="How subject relates to object, e.g. 'supports'."
    )
    object: str = pydantic.Field(
        description="The related concept, e.g. 'Incremental Processing'."
    )


class RelationshipList(pydantic.BaseModel):
    relationships: list[ExtractedRelationship] = pydantic.Field(default_factory=list)


SUMMARY_PROMPT = """\
You are an expert technical writer. Summarize the documentation below.
Return a concise title and a one-paragraph summary of what it covers.
"""

RELATIONSHIP_PROMPT = """\
You extract a concept knowledge graph from technical documentation.
List the salient (subject, predicate, object) relationships between concepts.
Focus on concepts and ignore code examples and implementation details.
Use concise noun phrases for subjects and objects and a short verb phrase for
the predicate. Each subject and object must name exactly one concept - never
a list of concepts joined by commas or "and". Return only relationships
supported by the text.
"""


# internal transfer type

@dataclass
class Triple:
    subject: str
    predicate: str
    object: str


@dataclass
class DocTriples:
    filename: str
    triples: list[Triple]


# llm summary extraction
@coco.fn(memo=True)
async def extract_summary(content: str) -> DocumentSummary:
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)
    result = await client.chat.completions.create(
        model=coco.use_context(LLM_MODEL),
        response_model=DocumentSummary,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    # re-validate to restore class identity for pickling (memo cache)
    return DocumentSummary.model_validate(result.model_dump())

# llm relationships extraction
@coco.fn(memo=True)
async def extract_relationships(content: str) -> list[Triple]:
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)
    result = await client.chat.completions.create(
        model=coco.use_context(LLM_MODEL),
        response_model=RelationshipList,
        messages=[
            {"role": "system", "content": RELATIONSHIP_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    validated = RelationshipList.model_validate(result.model_dump())
    triples = []
    for r in validated.relationships:
        if not r.subject or not r.object or "," in r.subject or "," in r.object:
            continue
        triples.append(Triple(r.subject, r.predicate, r.object))
    return triples


# per-file extraction, declare Document nodes, carry triplets forward
@coco.fn(memo=True)
async def process_file(file: localfs.File, document_table: neo4j.TableTarget[Document]) -> DocTriples:
    content = await file.read_text()
    filename = file.file_path.path.as_posix()
    
    summary = await extract_summary(content)
    document_table.declare_record(
        row=Document(filename=filename, title=summary.title, summary=summary.summary)
    )
    
    triples = await extract_relationships(content)
    return DocTriples(filename=filename, triples=triples)


# build the concept graph - Entity nodes and RELATIONSHIP / MENTION edges
@coco.fn
async def build_graph(
    docs: list[DocTriples],
    entity_table: neo4j.TableTarget[Entity],
    relationship_rel: neo4j.RelationTarget[Relationship],
    mention_rel: neo4j.RelationTarget[Any]
) -> None:
    entities: set[str] = set()
    mentions: set[tuple[str, str]] = set() # (filename, entity val)
    
    for doc in docs:
        for t in doc.triples:
            entities.add(t.subject)
            entities.add(t.object)
            mentions.add((doc.filename, t.subject))
            mentions.add((doc.filename, t.object))
            
            rel_id = await generate_id((t.subject, t.predicate, t.object))
            relationship_rel.declare_relation(
                from_id=t.subject,
                to_id=t.object,
                record=Relationship(id=rel_id, predicate=t.predicate),
            )
        
    for value in entities:
        entity_table.declare_record(row=Entity(value=value))
    
    for filename, entity in mentions:
        mention_rel.declare_relation(from_id=filename, to_id=entity)


# app main
@coco.fn
async def app_main(sourcedir: pathlib.Path) -> None:
    document_table = await neo4j.mount_table_target(
        KG_DB,
        "Document",
        await neo4j.TableSchema.from_class(Document, primary_key="filename"),
        primary_key="filename",
    )
    entity_table = await neo4j.mount_table_target(
        KG_DB,
        "Entity",
        await neo4j.TableSchema.from_class(Entity, primary_key="value"),
        primary_key="value",
    )
    
    # mount relation targets
    # RELATIONSHIP carries a predicate; mounted with a schema so each distinct
    # triple (keyed by the hashed `id`) becomes its own edge
    relationship_rel = await neo4j.mount_relation_target(
        KG_DB,
        "RELATIONSHIP",
        entity_table,
        entity_table,
        await neo4j.TableSchema.from_class(Relationship, primary_key="id"),
        primary_key="id",
    )
    # MENTION has no payload, the connector auto-derives the PK from endpoints
    mention_rel = await neo4j.mount_relation_target(
        KG_DB, "MENTION", document_table, entity_table
    )
    
    # per-file extraction
    files = localfs.walk_dir(
        sourcedir,
        recursive=True,
        path_matcher=PatternFilePathMatcher(included_patterns=["**/*.md", "**/*.mdx"]),
    )
    file_coros = []
    async for path_key, file in files.items():
        file_coros.append(
            coco.use_mount(
                coco.component_subpath("file", path_key),
                process_file,
                file,
                document_table,
            )
        )
    docs: list[DocTriples] = list(await asyncio.gather(*file_coros))
    
    # build the concept graph
    await coco.mount(
        coco.component_subpath("build_graph"),
        build_graph,
        docs,
        entity_table,
        relationship_rel,
        mention_rel
    )


app = coco.App(
    coco.AppConfig(name="DocsToKnowledgeGraph"),
    app_main,
    sourcedir=pathlib.Path("./dataset")
)
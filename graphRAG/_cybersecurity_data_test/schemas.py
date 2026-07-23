# neo4j node/edge dataclasses and LLM extraction schemas
from __future__ import annotations

from dataclasses import dataclass

import pydantic


@dataclass
class CveNode:
    cve_id: str
    description: str
    cvss_score: float
    published_date: str


@dataclass
class CweNode:
    cwe_id: str
    name: str


@dataclass
class ProductNode:
    key: str
    vendor: str
    product: str


@dataclass
class TechniqueNode:
    technique_id: str
    name: str
    tactic: str
    description: str


@dataclass
class MitigationNode:
    name: str


@dataclass
class HasWeakness:
    id: str


@dataclass
class Affects:
    id: str


@dataclass
class MitigatedBy:
    id: str


class ExtractedVulnerabilityType(pydantic.BaseModel):
    vulnerability_type: str = pydantic.Field(
        description=(
            "Short vulnerability class name derived from the description, "
            "e.g. 'remote code execution', 'privilege escalation', "
            "'server-side request forgery'."
        )
    )
    attack_vector_summary: str = pydantic.Field(
        description="One sentence describing how an attacker triggers this vulnerability."
    )


CVE_EXTRACTION_INSTRUCTION = """\
You are a cybersecurity analyst extracting structured information from a
CVE description. Identify the vulnerability class and summarize the
attack vector in one sentence, based only on the text provided. Then
prove your answer is correct by citing the source document and providing
it's title in one sentence.
"""
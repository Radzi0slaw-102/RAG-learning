"""Normalize raw downloaded data into the input shape the ingestion flows expect.

Reads:
  - nvd_CVE-*.json -> one file per CVE, raw NVD API 2.0 response shape
  - attack_stix.json -> full Enterprise ATT&CK STIX 2.1 bundle
  - kev_mapping.json -> raw Center for Threat-Informed Defense KEV-to-ATT&CK mapping

Writes:
  - cve_records.json
  - attack_techniques.json
  - kev_attack_mapping.json

Only techniques referenced by the KEV mapping subset are kept in
attack_techniques.json, since the full STIX bundle covers everything in
ATT&CK and only the ones relevant to our CVE sample are needed.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib


def normalize_cve(raw: dict) -> dict:
    cve = raw["vulnerabilities"][0]["cve"]

    description = next(
        (d["value"] for d in cve["descriptions"] if d["lang"] == "en"),
        cve["descriptions"][0]["value"],
    )

    cvss_score = None
    for metric_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metrics = cve.get("metrics", {}).get(metric_key)
        if metrics:
            primary = next((m for m in metrics if m.get("type") == "Primary"), metrics[0])
            cvss_score = primary["cvssData"]["baseScore"]
            break

    cwe_ids: list[str] = []
    for weakness in cve.get("weaknesses", []):
        for d in weakness.get("description", []):
            if d["lang"] == "en" and d["value"].startswith("CWE-") and d["value"] not in cwe_ids:
                cwe_ids.append(d["value"])

    affected_products = []
    seen = set()
    for source in cve.get("affected", []):
        for entry in source.get("affectedData", []):
            key = (entry["vendor"], entry["product"])
            if key in seen:
                continue
            seen.add(key)
            affected_products.append({"vendor": entry["vendor"], "product": entry["product"]})

    return {
        "cve_id": cve["id"],
        "description": description,
        "cvss_score": cvss_score if cvss_score is not None else 0.0,
        "published_date": cve["published"][:10],
        "cwe_ids": cwe_ids,
        "affected_products": affected_products,
    }


def load_stix_indexes(bundle: dict) -> tuple[dict, dict, list]:
    """Return (attack_pattern by id, course_of_action by id, relationships)."""
    patterns, mitigations, relationships = {}, {}, []
    for obj in bundle["objects"]:
        obj_type = obj.get("type")
        if obj_type == "attack-pattern":
            patterns[obj["id"]] = obj
        elif obj_type == "course-of-action":
            mitigations[obj["id"]] = obj
        elif obj_type == "relationship":
            relationships.append(obj)
    return patterns, mitigations, relationships


def external_id(obj: dict) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def normalize_attack_techniques(bundle: dict, technique_ids: set[str]) -> list[dict]:
    patterns, mitigations, relationships = load_stix_indexes(bundle)

    by_technique_id = {}
    for stix_id, pattern in patterns.items():
        tid = external_id(pattern)
        if tid in technique_ids:
            by_technique_id[tid] = (stix_id, pattern)

    mitigates_by_pattern: dict[str, list[str]] = {}
    for rel in relationships:
        if rel.get("relationship_type") != "mitigates" or rel.get("revoked"):
            continue
        target_pattern = patterns.get(rel["target_ref"])
        source_mitigation = mitigations.get(rel["source_ref"])
        if not target_pattern or not source_mitigation:
            continue
        tid = external_id(target_pattern)
        if tid not in technique_ids:
            continue
        mitigates_by_pattern.setdefault(tid, []).append(source_mitigation["name"])

    results = []
    for tid, (stix_id, pattern) in by_technique_id.items():
        tactic = pattern.get("kill_chain_phases", [{}])[0].get("phase_name", "unknown")
        results.append({
            "technique_id": tid,
            "name": pattern["name"],
            "tactic": tactic.replace("-", " ").title(),
            "description": pattern.get("description", "").split("\n")[0][:500],
            "mitigations": mitigates_by_pattern.get(tid, []),
        })
    return results


def normalize_kev_mapping(raw: dict, cve_filter: set[str] | None) -> list[dict]:
    objects = raw["mapping_objects"]
    if cve_filter:
        objects = [m for m in objects if m["capability_id"] in cve_filter]
    return [
        {
            "capability_id": m["capability_id"],
            "capability_description": m["capability_description"],
            "mapping_type": m["mapping_type"],
            "attack_object_id": m["attack_object_id"],
            "attack_object_name": m["attack_object_name"],
        }
        for m in objects
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvd-dir", required=True, help="Directory containing nvd_CVE-*.json files")
    parser.add_argument("--attack-stix", required=True, help="Path to the full ATT&CK STIX bundle JSON")
    parser.add_argument("--kev-mapping", required=True, help="Path to the raw KEV-to-ATT&CK mapping JSON")
    parser.add_argument("--cve-filter", default="", help="Comma-separated CVE IDs to keep from the KEV mapping")
    args = parser.parse_args()

    cve_filter = {c.strip() for c in args.cve_filter.split(",") if c.strip()} or None

    nvd_files = sorted(glob.glob(str(pathlib.Path(args.nvd_dir) / "nvd_CVE-*.json")))
    cve_records = [normalize_cve(json.loads(open(f).read())) for f in nvd_files]
    with open("cve_records.json", "w") as f:
        json.dump({"cve_records": cve_records}, f, indent=2)
    print(f"Wrote {len(cve_records)} CVE records to cve_records.json")

    kev_raw = json.loads(open(args.kev_mapping).read())
    mapping_objects = normalize_kev_mapping(kev_raw, cve_filter)
    with open("kev_attack_mapping.json", "w") as f:
        json.dump({"mapping_objects": mapping_objects}, f, indent=2)
    print(f"Wrote {len(mapping_objects)} mapping edges to kev_attack_mapping.json")

    technique_ids = {m["attack_object_id"] for m in mapping_objects}
    stix_bundle = json.loads(open(args.attack_stix).read())
    techniques = normalize_attack_techniques(stix_bundle, technique_ids)
    with open("attack_techniques.json", "w") as f:
        json.dump({"techniques": techniques}, f, indent=2)
    print(f"Wrote {len(techniques)} techniques to attack_techniques.json")

    missing = technique_ids - {t["technique_id"] for t in techniques}
    if missing:
        print(f"WARNING: techniques referenced by the mapping but not found in the STIX bundle: {missing}")


if __name__ == "__main__":
    main()
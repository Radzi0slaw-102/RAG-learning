import argparse
import csv
import os
import time
import urllib.parse
import urllib.request
import json

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PAGE_SIZE = 100


def fetch_cves(keyword: str, limit: int) -> list[dict]:
    results = []
    start_index = 0
    while len(results) < limit:
        params = {
            "keywordSearch": keyword,
            "resultsPerPage": min(PAGE_SIZE, limit - len(results)),
            "startIndex": start_index,
        }
        url = f"{NVD_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url)
        
        with urllib.request.urlopen(req) as resp:
            payload = json.load(resp)
        
        page = payload.get("vulnerabilities", [])
        if not page:
            break
        results.extend(page)
        start_index += len(page)
        
        time.sleep(6.0)
        
    return results[:limit]


def describe_cpe(match: dict) -> str:
    criteria = match.get("criteria", "")
    # cpe:2.3:a:vendor:product:version:...
    parts = criteria.split(":")
    if len(parts) >= 6:
        vendor, product, version = parts[3], parts[4], parts[5]
        return f"{vendor}/{product} {version}".replace("*", "any")
    return criteria


def format_document(item: dict) -> tuple[str, str, str]:
    cve = item["cve"]
    cve_id = cve["id"]
    
    descriptions = cve.get("descriptions", [])
    text_en = next((d["value"] for d in descriptions if d["lang"] == "en"), "")
    
    cpe_lines = []
    for config in cve.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                if match.get("vulnerable"):
                    cpe_lines.append(describe_cpe(match))
    cpe_lines = sorted(set(cpe_lines))
    
    metrics = cve.get("metrics", {})
    severity = ""
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics:
            severity = metrics[key][0]["cvssData"].get("baseSeverity", "")
            break
    
    title = f"{cve_id}: {text_en[:80]}"
    body_lines = [f"# {cve_id}", "", text_en, ""]
    if severity:
        body_lines.append(f"Severity: {severity}")
    if cpe_lines:
        body_lines.append("Affected packages:")
        body_lines.extend(f"- {line}" for line in cpe_lines)
    body = "\n".join(body_lines)
    
    return title, body, cve_id


def write_dataset(items: list[dict], out_dir: str) -> None:
    docs_dir = os.path.join(out_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "text", "cve_id"])
        for item in items:
            title, body, cve_id = format_document(item)
            with open(os.path.join(docs_dir, f"{cve_id}.md"), "w", encoding="utf-8") as doc_f:
                doc_f.write(body)
            writer.writerow([title, body, cve_id])
    
    print(f"Wrote {len(items)} documents to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyword", required=True, help="NVD keyword search, e.g. 'lodash' or 'django'")
    parser.add_argument("--limit", type=int, default=50, help="Max number of CVE records")
    parser.add_argument("--out", required=True, help="Output dataset directory, e.g. datasets/npm_lodash")
    parser.add_argument("--api-key", default=os.environ.get("NVD_API_KEY"))
    args = parser.parse_args()
    
    items = fetch_cves(args.keyword, args.limit, args.api_key)
    write_dataset(items, args.out)


if __name__ == "__main__":
    main()
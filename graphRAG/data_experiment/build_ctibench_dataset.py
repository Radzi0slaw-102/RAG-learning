"""
Build a canonical dataset and question set from CTI-Bench's cti-rcm subset
(CVE description -> CWE ground truth), instead of hand-written questions.

Unlike download_cve.py, this doesn't hit NVD - it pulls a published,
peer-reviewed benchmark (CTIBench, NeurIPS'24) with existing ground truth,
so no must_contain values need to be guessed or manually verified.

Each row's raw CVE description becomes one document; the corresponding
CWE ID becomes the question's expected answer. This produces "situational"
document content (the description never states its own CWE ID), so a
correct answer still requires the method to classify the described
behavior, not just echo text back.

Output layout matches the canonical dataset format:
    datasets/<name>/manifest.csv   -- columns: title,text,cve_id
    datasets/<name>/docs/<row_id>.md
    eval/questions_<name>.yaml

Usage:
    python data/build_ctibench_dataset.py --subset cti-rcm --limit 30 --out datasets/ctibench_rcm
"""
import argparse
import csv
import os
import re

import yaml
from datasets import load_dataset

HF_REPO = "AI4Sec/cti-bench"

QUESTION_TEMPLATES = {
    "cti-rcm": "A system log shows the following vulnerability behavior: {description} What weakness category (CWE) does this correspond to?",
    "cti-vsp": "A system log shows the following vulnerability behavior: {description} What CVSS base score would this vulnerability receive?",
}


def extract_cve_id(url: str, prompt: str) -> str:
    match = re.search(r"CVE-\d{4}-\d{4,7}", url) or re.search(r"CVE-\d{4}-\d{4,7}", prompt)
    return match.group(0) if match else "UNKNOWN"


INSTRUCTION_MARKERS = ["Analyze the following", "Given the following", "Based on the description above"]


def strip_instruction_suffix(prompt: str) -> str:
    # some CTI-Bench subsets append LLM-formatting instructions after the raw
    # description; cti-rcm's Prompt field has matched the raw CVE description
    # verbatim in samples checked so far, but this is a defensive fallback
    for marker in INSTRUCTION_MARKERS:
        idx = prompt.find(marker)
        if idx > 0:
            return prompt[:idx].strip()
    return prompt.strip()


def build(subset: str, limit: int, out_dir: str) -> None:
    dataset = load_dataset(HF_REPO, subset, split="test")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    docs_dir = os.path.join(out_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    manifest_path = os.path.join(out_dir, "manifest.csv")
    questions = []

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "text", "cve_id"])

        for i, row in enumerate(dataset):
            description = strip_instruction_suffix(row["Prompt"])
            cve_id = extract_cve_id(row["URL"], description)
            doc_id = cve_id if cve_id != "UNKNOWN" else f"row{i:04d}"
            title = f"{doc_id}: {description[:80]}"

            with open(os.path.join(docs_dir, f"{doc_id}.md"), "w", encoding="utf-8") as doc_f:
                doc_f.write(f"# {doc_id}\n\n{description}\n")
            writer.writerow([title, description, cve_id])

            question_text = QUESTION_TEMPLATES[subset].format(description=description)
            questions.append({
                "id": f"ctibench_{i:04d}",
                "question": question_text,
                "must_contain": [row["GT"]],
            })

    questions_path = f"eval/questions_{os.path.basename(out_dir.rstrip('/'))}.yaml"
    with open(questions_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"questions": questions}, f, sort_keys=False, allow_unicode=True)

    print(f"Wrote {len(questions)} documents to {out_dir}")
    print(f"Wrote {len(questions)} questions to {questions_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=list(QUESTION_TEMPLATES), default="cti-rcm")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out", required=True, help="Output dataset directory, e.g. datasets/ctibench_rcm")
    args = parser.parse_args()

    build(args.subset, args.limit, args.out)


if __name__ == "__main__":
    main()
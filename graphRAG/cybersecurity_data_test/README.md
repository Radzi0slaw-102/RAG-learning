# Requirements
- Packages
```bash
# using python 3.11.9 env
pip install cocoindex instructor litellm pydantic neo4j
```
- Local LLM
```bash
ollama pull llama3.1:8b
```
- Docker

# How to run

## Download raw data
```bash
# in ./data/raw directory

# find yours CVE documents that you see fit
for cve in CVE-2021-44228 CVE-2014-6271 CVE-2021-26855 CVE-2019-19781 CVE-2020-1472; do
  curl -s "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=$cve" -o "nvd_${cve}.json"
  sleep 6
done
curl -s "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack-15.1.json" -o attack_stix.json
curl -s "https://center-for-threat-informed-defense.github.io/mappings-explorer/data/kev/attack-15.1/kev-02.13.2025/enterprise/kev-02.13.2025_attack-15.1-enterprise_json.json" -o kev_mapping.json
```
## Create .env in main directory
```bash
# example content
COCOINDEX_DB=./cocoindex.db

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=cocoindex
NEO4J_DATABASE=neo4j

LLM_MODEL=ollama/llama3.1:8b
```
## Normalize data to fit it into evaluation process
```bash
# in ./data directory
# change all CVE documents to your chosen names
python normalize_raw_data.py --nvd-dir raw --attack-stix raw/attack_stix.json --kev-mapping raw/kev_mapping.json --cve-filter CVE-2021-44228,CVE-2014-6271,CVE-2021-26855,CVE-2019-19781,CVE-2020-1472
```
## Setup Docker
```bash
docker compose up -d
```
## Run cocoindex apps
```bash
cocoindex update cve_flow
cocoindex update attack_flow
cocoindex update link_flow
```
You can now lookup for database structure on localhost:7474
## Generate quesitons and evaluate
```bash
python generate_questions.py
python evaluate.py
```
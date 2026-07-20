# How to use
By default, scripts use `./config/models.yaml` configuration file. You can change it for your own needs, just make sure to update line in `rag_pipeline.py`:
```python
CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
```
with desired filename.

Then, run files in src subfolder in order:
 - `ingest.py`
 - `rag_pipeline.py`
 - `run_pipeline.py`
 - `evaluate.py`

Results are written to:
 - `./results/raw`
 - `./data`
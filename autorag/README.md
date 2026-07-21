# How to run
1. To run "just" the AutoRAG evaluation:
```sh
autorag evaluate --config ./configs/config_big.yaml --qa_data_path ./data/qa.parquet --corpus_data_path ./data/corpus.parquet
```
OR for micro version:
```sh
autorag evaluate --config ./configs/config_small.yaml --qa_data_path ./data/qa_micro.parquet --corpus_data_path ./data/corpus_micro.parquet
```

2. To run with system performance monitor:
```sh
python run_benchmark.py --config ./configs/config_big.yaml --qa_data_path ./data/qa.parquet --corpus_data_path ./data/corpus.parquet
```
OR for micro version:
```sh
python run_benchmark.py --config ./configs/config_small.yaml --qa_data_path ./data/qa_micro.parquet --corpus_data_path ./data/corpus_micro.parquet
```
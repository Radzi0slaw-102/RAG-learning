import sys
sys.path.insert(0, "src")

from pipeline import build_index, run_query
from config import DATA_URL

if __name__ == "__main__":
    index, llm = build_index(DATA_URL, force_rebuild=True)
    answer = run_query(index, llm, "What are news related to financial sector?")
    print(answer)
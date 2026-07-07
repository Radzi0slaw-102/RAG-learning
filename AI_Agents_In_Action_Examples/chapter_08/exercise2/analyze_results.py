"""
Parser and analyzer for semantic search results log (TF-IDF / Word2Vec / BERT API).

Writes the analysis to results_analyze.txt (current directory), containing:
    - per-query table: best match and score for each method, search time
    - aggregate stats: avg/min/max score per method
    - ranking agreement (checks if methods pointet to the same top sentence)
    - time summary
"""

import re
import sys
import statistics
from dataclasses import dataclass, field

@dataclass
class MethodResult:
    matches: list # list of (score: float, text: str)
    search_time: float # seconds

@dataclass
class QueryResult:
    query: str
    methods: dict = field(default_factory=dict) # method_name -> MethodResult

METHOD_NAMES = ["TF-IDF", "Word2Vec", "BERT API"]


def parse_log(text: str):
    queries = []
    # split file in blocks per query
    query_blocks = re.split(r"=== QUERY: (.+?) ===", text)
    
    # skip query_blocks[0], as it's build times
    for i in range(1, len(query_blocks), 2):
        query_name = query_blocks[i].strip()
        block = query_blocks[i + 1]
        
        qr = QueryResult(query=query_name)
        
        # for each method it's match and time section is withdrawn
        for idx, method in enumerate(METHOD_NAMES):
            pattern = re.escape(method) + r" top matches:\s*\n((?:  - Score: .+\n)+)Local query search time: ([\d.]+)"
            m = re.search(pattern, block)
            if not m:
                continue
            matches_block, search_time = m.groups()
            matches = []
            
            for line in matches_block.strip().splitlines():
                mm = re.match(r"\s*- Score: ([\d.]+) \| (.+)", line)
                if mm:
                    score, doc_text = mm.groups()
                    matches.append((float(score), doc_text.strip()))
            
            qr.methods[method] = MethodResult(matches=matches, search_time=float(search_time))
        
        if qr.methods:
            queries.append(qr)
            
    # summary from the end of the file
    summary = {}
    summary_match = re.search(r"=== Time performance summary ===\s*\n((?:.+\n?)+)", text)
    if summary_match:
        for line in summary_match.group(1).strip().splitlines():
            m = re.match(r"(.+?) - Avg search time: ([\d.]+) ms", line)
            if m:
                summary[m.group(1).strip()] = float(m.group(2))
    
    # build times
    build_times = {}
    for m in re.finditer(r"(TF-IDF|Word2Vec|BERT API) database build time: ([\d.]+) seconds", text):
        build_times[m.group(1)] = float(m.group(2))
    
    return queries, summary, build_times
    

def analyze(queries, summary, build_times, out):
    def w(line=""):
        print(line, file=out)
    
    w("-" * 67)
    w("I. Base building times")
    w("-" * 67)
    for method, t in build_times.items():
        w(f"  {method:12s}: {t*1000:.2f} ms")
        
    w()
    w("-" * 67)
    w(f"II. Per-query details  (total: {len(queries)} queries)")
    w("-" * 67)
    
    top_scores = {m: [] for m in METHOD_NAMES}
    search_times = {m: [] for m in METHOD_NAMES}
    agreement_tfidf_bert = 0
    agreement_w2v_bert = 0
    zero_score_queries = {m: 0 for m in METHOD_NAMES}
    
    for q in queries:
        w(f"\n[{q.query}]")
        top_docs = {}
        for method in METHOD_NAMES:
            if method not in q.methods:
                continue
            
            mr = q.methods[method]
            best_score, best_doc = mr.matches[0] if mr.matches else (0.0, "-")
            
            top_docs[method] = best_doc
            top_scores[method].append(best_score)
            search_times[method].append(mr.search_time * 1000)
            
            if best_score == 0.0:
                zero_score_queries[method] += 1
            w(f"  {method:10s} top1: {best_score:.4f} | {best_doc[:55]:55s} | {mr.search_time*1000:.2f} ms")
        
        if top_docs.get("TF-IDF") == top_docs.get("BERT API"):
            agreement_tfidf_bert += 1
        if top_docs.get("Word2Vec") == top_docs.get("BERT API"):
            agreement_w2v_bert += 1
        
    w()
    w("-" * 67)
    w("III. Aggregate quality statistics (best score per query)")
    w("-" * 67)
    for method in METHOD_NAMES:
        scores = top_scores[method]
        if not scores:
            continue
        w(f"  {method:10s}: avg={statistics.mean(scores):.4f}  "
          f"min={min(scores):.4f}  max={max(scores):.4f}  "
          f"queries with score=0: {zero_score_queries[method]}/{len(queries)}")
    
    w()
    w("-" * 67)
    w("IV. Aggregate search time statistics")
    w("-" * 67)
    for method in METHOD_NAMES:
        times = search_times[method]
        if not times:
            continue
        w(f"  {method:10s}: avg={statistics.mean(times):.2f} ms  "
          f"min={min(times):.2f} ms  max={max(times):.2f} ms")
    
    if summary:
        w()
        w("  (Summary as reported by the script itself):")
        for method, t in summary.items():
            w(f"    {method:10s}: {t:.2f} ms")
 
    w()
    w("-" * 67)
    w("V. Ranking agreement (does the best sentences match)")
    w("-" * 67)
    n = len(queries)
    w(f"  TF-IDF   vs BERT API: {agreement_tfidf_bert}/{n} queries ({agreement_tfidf_bert/n*100:.1f}%)")
    w(f"  Word2Vec vs BERT API: {agreement_w2v_bert}/{n} queries ({agreement_w2v_bert/n*100:.1f}%)")
    
    w()
    w("-" * 67)
    w("VI. Quick takeaway (quality vs. time cost)")
    w("-" * 67)
    for method in METHOD_NAMES:
        if not top_scores[method] or not search_times[method]:
            continue
        avg_score = statistics.mean(top_scores[method])
        avg_time = statistics.mean(search_times[method])
        w(f"  {method:10s}: quality={avg_score:.3f}  time={avg_time:.2f} ms  "
          f"quality/ms={avg_score/avg_time:.5f}")
 
 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <log_file>")
        sys.exit(1)
 
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        content = f.read()
 
    queries, summary, build_times = parse_log(content)
 
    output_path = "results_analyze.txt"
    with open(output_path, "w", encoding="utf-8") as out:
        analyze(queries, summary, build_times, out)
 
    print(f"Analysis written to {output_path}")
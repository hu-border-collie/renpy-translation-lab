#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG Store Performance Benchmark Script
This script measures load, search, and upsert performance of JsonRagStore
at synthetic scales (e.g. 100, 1000, 10000 records) offline.
It prints results in a markdown table and outputs dynamic optimization recommendations.
"""

import argparse
import os
import random
import math
import time
import tempfile
from datetime import datetime

# Import JsonRagStore and utility functions from the local directory
try:
    from rag_memory import JsonRagStore, hash_text
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from rag_memory import JsonRagStore, hash_text


def random_unit_vector(dim):
    """
    Generate a random unit vector of dimension `dim`.
    Unit vectors have a norm of 1.0, which is standard for embeddings.
    """
    vec = [random.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        return [x / norm for x in vec]
    return [0.0] * dim


def generate_synthetic_records(count, dim):
    """
    Generate synthetic records with unique IDs and realistic text lengths.
    """
    records = []
    quality_states = ['seed', 'batch_applied', 'revision_applied', 'sync_applied', 'manual_polished']
    
    for i in range(count):
        # Unique source text based on index to avoid duplicate hash/memory_id
        source_text = f"Synthetic source text number {i} with some extra padding to make it a realistic sentence length. Let's add more characters here to simulate standard game script strings."
        translated_text = f"合成译文，编号为 {i} 的段落，同样带有足够的中文填充字符，以使其达到合理的翻译句子长度。"
        
        # Memory ID should be unique, matching how real code uses hash_text
        memory_id = hash_text(source_text)
        
        records.append({
            'memory_id': memory_id,
            'file_rel_path': f"game/chapter_{random.randint(1, 10)}.rpy",
            'source_text': source_text,
            'translated_text': translated_text,
            'embedding': random_unit_vector(dim),
            'quality_state': random.choice(quality_states),
            'created_at': datetime.now().isoformat(timespec='seconds'),
        })
    return records


def run_benchmark_for_size(size, queries_count, dim):
    """
    Run the full suite of benchmarks for a specific scale (database size).
    """
    # 1. Generate synthetic data
    records = generate_synthetic_records(size, dim)
    
    # Generate incremental update batch (10 items)
    incremental_records = generate_synthetic_records(10, dim)
    # Ensure they have unique text so they don't overlap with main records
    for i, rec in enumerate(incremental_records):
        rec['source_text'] += f" _incremental_{size}_{i}"
        rec['memory_id'] = hash_text(rec['source_text'])
    
    results = {}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 2. Benchmark Initial Bulk Upsert
        store = JsonRagStore(tmp_dir)
        t_start = time.perf_counter()
        changed = store.upsert_history(records)
        t_end = time.perf_counter()
        
        results['bulk_upsert_s'] = t_end - t_start
        assert changed == size, f"Expected {size} records written, got {changed}"
        
        # 3. Benchmark File Size
        history_path = os.path.join(tmp_dir, 'history.jsonl')
        if os.path.isfile(history_path):
            file_size_bytes = os.path.getsize(history_path)
            results['file_size_mb'] = file_size_bytes / (1024.0 * 1024.0)
        else:
            results['file_size_mb'] = 0.0
            
        # 4. Benchmark Store Load (new instance)
        load_store = JsonRagStore(tmp_dir)
        t_start = time.perf_counter()
        load_store.load()
        t_end = time.perf_counter()
        results['load_s'] = t_end - t_start
        assert load_store.count_history() == size, f"Expected {size} records loaded, got {load_store.count_history()}"
        
        # 5. Benchmark Search Queries
        # We test 3 query modes:
        # A: Zero-Hit (random vector, default min_similarity=0.72)
        # B: Exact-Hit (existing vector from db, default min_similarity=0.72)
        # C: All-Match (random vector, min_similarity=-1.0 to force sorting and slicing of all items)
        
        # A: Zero-Hit Search
        zero_hit_queries = [random_unit_vector(dim) for _ in range(queries_count)]
        t_start = time.perf_counter()
        for q_vec in zero_hit_queries:
            load_store.search_history(q_vec, top_k=4, min_similarity=0.72)
        t_end = time.perf_counter()
        results['search_zero_hit_ms'] = ((t_end - t_start) / queries_count) * 1000.0
        
        # B: Exact-Hit Search (Cache Hit)
        # Select embeddings from the database
        sample_records = random.sample(records, min(queries_count, size))
        exact_vectors = [rec['embedding'] for rec in sample_records]
        # Pad with random unit vectors if we have fewer sample records than queries_count
        while len(exact_vectors) < queries_count:
            exact_vectors.append(random.choice(records)['embedding'])
            
        t_start = time.perf_counter()
        for q_vec in exact_vectors:
            hits = load_store.search_history(q_vec, top_k=4, min_similarity=0.72)
            # Sanity check: must match at least 1 record (the source record itself, similarity near 1.0)
            assert len(hits) >= 1, "Cache hit search should return at least one match"
        t_end = time.perf_counter()
        results['search_cache_hit_ms'] = ((t_end - t_start) / queries_count) * 1000.0
        
        # C: All-Match Search (Exhaustive sorting)
        all_match_queries = [random_unit_vector(dim) for _ in range(queries_count)]
        t_start = time.perf_counter()
        for q_vec in all_match_queries:
            hits = load_store.search_history(q_vec, top_k=4, min_similarity=-1.0)
            # Sanity check: should return exactly top_k (4) unless N < 4
            expected_hits = min(4, size)
            assert len(hits) == expected_hits, f"Expected {expected_hits} hits, got {len(hits)}"
        t_end = time.perf_counter()
        results['search_all_match_ms'] = ((t_end - t_start) / queries_count) * 1000.0
        
        # 6. Benchmark Incremental Write (10 new items)
        t_start = time.perf_counter()
        changed_inc = load_store.upsert_history(incremental_records)
        t_end = time.perf_counter()
        results['incremental_upsert_s'] = t_end - t_start
        assert changed_inc == 10, f"Expected 10 incremental updates, got {changed_inc}"
        
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run performance benchmark for Ren'Py Translation Lab RAG memory store."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="100,1000,10000",
        help="Comma-separated list of history database sizes to benchmark (default: 100,1000,10000)"
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=20,
        help="Number of query iterations to measure average search latency (default: 20)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=768,
        help="Dimensionality of embeddings (default: 768)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    
    # Parse and validate user-controlled benchmark dimensions.
    try:
        sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    except ValueError:
        parser.error("Invalid --sizes format. Please provide comma-separated integers.")
    if not sizes:
        parser.error("--sizes must contain at least one positive integer.")
    if any(size <= 0 for size in sizes):
        parser.error("--sizes values must all be positive integers.")
    if args.queries <= 0:
        parser.error("--queries must be a positive integer.")
    if args.dim <= 0:
        parser.error("--dim must be a positive integer.")
        
    print(f"============================================================")
    print(f"RAG Store Performance Benchmark")
    print(f"============================================================")
    print(f"Parameters:")
    print(f"  - Database sizes: {sizes}")
    print(f"  - Query iterations: {args.queries}")
    print(f"  - Embedding dimension: {args.dim}")
    print(f"  - Random seed: {args.seed}")
    print(f"============================================================")
    print("Running benchmarks... (this might take a few moments)")
    
    all_results = {}
    for size in sizes:
        print(f"Benchmarking scale N = {size}...")
        all_results[size] = run_benchmark_for_size(size, args.queries, args.dim)
        
    print("\n")
    print(f"### Benchmark Results (Embedding Dim: {args.dim}, Queries per test: {args.queries})")
    print(f"| Scale (N) | Bulk Upsert (s) | Load Store (s) | Zero-Hit Search (ms) | Cache-Hit Search (ms) | All-Match Search (ms) | Incremental Upsert (s) | File Size (MB) |")
    print(f"|---|---|---|---|---|---|---|---|")
    
    for size in sorted(sizes):
        res = all_results[size]
        print(f"| {size:<9} | {res['bulk_upsert_s']:<15.4f} | {res['load_s']:<14.4f} | {res['search_zero_hit_ms']:<20.2f} | {res['search_cache_hit_ms']:<21.2f} | {res['search_all_match_ms']:<21.2f} | {res['incremental_upsert_s']:<22.4f} | {res['file_size_mb']:<14.2f} |")
        
    print("\n### Recommendations and Threshold Analysis")
    
    # Analyze the largest scale tested to suggest threshold alerts
    max_size = max(sizes)
    max_res = all_results[max_size]
    
    # Threshold Constants (defined for warning triggers)
    SEARCH_SLOW_THRESHOLD_MS = 50.0
    UPSERT_SLOW_THRESHOLD_S = 1.0
    
    # Check Search latency recommendations
    search_worst_ms = max(max_res['search_zero_hit_ms'], max_res['search_cache_hit_ms'], max_res['search_all_match_ms'])
    print(f"- **Search Latency (at scale {max_size})**:")
    print(f"  - Measured max average search time: {search_worst_ms:.2f} ms (Warning threshold: {SEARCH_SLOW_THRESHOLD_MS} ms)")
    if search_worst_ms > SEARCH_SLOW_THRESHOLD_MS:
        print(f"  - [WARNING] Search latency is high. It is recommended to implement:")
        print(f"    1. **Norm Caching**: Pre-calculate and store the vector norms to avoid computing them inside the cosine similarity loop.")
        print(f"    2. **NumPy Vectorization**: Replace the pure-Python cosine similarity loop with vectorized NumPy operations to handle high-dimensional computations.")
    else:
        print(f"  - [OK] Search latency is within acceptable limits. Pure-Python linear scan is sufficient for this scale.")
        
    # Check Upsert latency recommendations
    inc_upsert_s = max_res['incremental_upsert_s']
    print(f"- **Incremental Upsert Latency (at scale {max_size})**:")
    print(f"  - Measured write time for 10 records: {inc_upsert_s:.4f} s (Warning threshold: {UPSERT_SLOW_THRESHOLD_S} s)")
    if inc_upsert_s > UPSERT_SLOW_THRESHOLD_S:
        print(f"  - [WARNING] Upsert latency is high. This is because the database is completely sorted and rewritten on every write.")
        print(f"    - Recommendation: Switch from a single atomic-rewritten JSONL file to an **append-only log** with periodic **compaction**, or migrate to a lightweight database engine like **SQLite**.")
    else:
        print(f"  - [OK] Write latency is within acceptable limits. Atomic-replace JSONL is safe and reliable for this scale.")
        
    print("\nBenchmark completed successfully.")


if __name__ == '__main__':
    main()

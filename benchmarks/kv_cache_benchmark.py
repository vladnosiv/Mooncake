#!/usr/bin/env python3
"""
KV-Cache Benchmark for MoonCake Store

Benchmarks MoonCake Store performance with simulated KV-cache workload.
Generates shared prefix patterns similar to inference workloads and measures
cache hit rates, throughput, and latency.

Usage:
python3 benchmarks/kv_cache_benchmark.py --master-server 127.0.0.1:50051

python3 benchmarks/kv_cache_benchmark.py --dataset-name generated-shared-prefix \
    --block-size-tokens 256 --token-size-kb 160 \
    --gsp-num-groups 64 --gsp-prompts-per-group 16 \
    --global-segment-size 3200 --local-segment-size 512
"""

import argparse
import asyncio
import hashlib
import json
import os
import pickle
import random
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.asyncio import tqdm

from mooncake.store import MooncakeDistributedStore

@dataclass
class KVCacheRequest:
    """Represents a single KV-cache request with block hashes."""
    request_id: str
    block_hashes: List[str]  # List of block hashes for this request
    block_sizes: List[int]   # Size of each block in bytes
    num_tokens: int          # Total number of tokens
    prefix_length: int       # Length of shared prefix (for metrics)

@dataclass 
class RequestResult:
    """Result of processing a single request."""
    request_id: str
    success: bool = False
    latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    bytes_retrieved: int = 0
    error: str = ""
    
    # Detailed timing
    exist_check_time: float = 0.0
    get_data_time: float = 0.0

@dataclass
class BenchmarkMetrics:
    """Overall benchmark metrics."""
    completed: int
    total_requests: int
    total_blocks_checked: int
    total_cache_hits: int
    total_cache_misses: int
    total_bytes_retrieved: int
    request_throughput: float
    block_throughput: float
    cache_hit_rate: float
    mean_latency_ms: float
    median_latency_ms: float
    p99_latency_ms: float
    mean_exist_check_ms: float
    mean_get_data_ms: float
    duration: float


def generate_block_hash(block_id: int, content: bytes = None) -> str:
    """Generate a deterministic hash for a block."""
    if content is None:
        # Generate deterministic content based on block_id
        content = f"block_{block_id}_".encode() * (256 // len(f"block_{block_id}_") + 1)
    
    return hashlib.sha256(content).hexdigest()


def generate_block_data(block_id: int, size_bytes: int) -> bytes:
    """Generate deterministic block data."""
    # Use block_id as seed for reproducible data
    random_gen = random.Random(block_id)
    return bytes(random_gen.randint(0, 255) for _ in range(size_bytes))


def get_gen_prefix_cache_path(args) -> Path:
    """Create cache directory for generated data."""
    cache_dir = Path.home() / ".cache" / "mooncake" / "benchmark" 
    cache_key = (
        f"kv_cache_{args.gsp_num_groups}_{args.gsp_prompts_per_group}_"
        f"{args.gsp_system_prompt_len}_{args.gsp_question_len}_"
        f"{args.block_size_tokens}_{args.token_size_kb}.pkl"
    )
    return cache_dir / cache_key


def sample_generated_shared_prefix_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    block_size_tokens: int,
    token_size_kb: int,
    args: argparse.Namespace,
) -> List[KVCacheRequest]:
    """Generate KV-cache requests with shared system prefixes."""
    
    cache_path = get_gen_prefix_cache_path(args)
    
    # Try to load from cache first
    if cache_path.exists():
        print(f"\nLoading cached generated data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    print("\nGenerating new KV-cache request data...")
    
    token_size_bytes = token_size_kb * 1024
    
    # Generate system prompt blocks for each group
    system_prompt_blocks = []
    for group_idx in range(num_groups):
        num_blocks = (system_prompt_len + block_size_tokens - 1) // block_size_tokens
        group_blocks = []
        
        for block_idx in range(num_blocks):
            # Create unique block ID that ensures shared prefixes across group
            block_id = f"system_g{group_idx}_b{block_idx}"
            block_hash = generate_block_hash(hash(block_id), block_id.encode())
            group_blocks.append((block_hash, token_size_bytes))
        
        system_prompt_blocks.append(group_blocks)
    
    # Generate requests
    input_requests = []
    total_blocks_generated = 0
    
    for group_idx in tqdm(range(num_groups), desc="Generating groups"):
        system_blocks = system_prompt_blocks[group_idx]
        
        for prompt_idx in tqdm(range(prompts_per_group), desc="Generating requests", leave=False):
            # Generate question blocks (unique per request)
            num_question_blocks = (question_len + block_size_tokens - 1) // block_size_tokens
            question_blocks = []
            
            for block_idx in range(num_question_blocks):
                block_id = f"question_g{group_idx}_p{prompt_idx}_b{block_idx}"
                block_hash = generate_block_hash(hash(block_id), block_id.encode())
                question_blocks.append((block_hash, token_size_bytes))
            
            # Combine system and question blocks
            all_blocks = system_blocks + question_blocks
            block_hashes = [h for h, _ in all_blocks]
            block_sizes = [s for _, s in all_blocks]
            
            request = KVCacheRequest(
                request_id=f"req_g{group_idx}_p{prompt_idx}",
                block_hashes=block_hashes,
                block_sizes=block_sizes,
                num_tokens=system_prompt_len + question_len,
                prefix_length=system_prompt_len,
            )
            input_requests.append(request)
            total_blocks_generated += len(block_hashes)
    
    # Shuffle requests to simulate real workload
    random.shuffle(input_requests)
    
    # Print statistics
    print(f"\nGenerated KV-cache dataset statistics:")
    print(f"Number of groups: {num_groups}")
    print(f"Prompts per group: {prompts_per_group}")
    print(f"Total requests: {len(input_requests)}")
    print(f"Total blocks generated: {total_blocks_generated}")
    print(f"Block size: {block_size_tokens} tokens ({token_size_kb} KB)")
    print(f"Average blocks per request: {total_blocks_generated / len(input_requests):.1f}")
    print(f"System prefix length: {system_prompt_len} tokens")
    print(f"Question length: {question_len} tokens\n")
    
    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Caching generated data to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(input_requests, f)
    
    return input_requests


def setup_store_client(args) -> MooncakeDistributedStore:
    """Initialize MoonCake Store client."""
    store = MooncakeDistributedStore()
    
    global_segment_size = args.global_segment_size * 1024 * 1024  # Convert MB to bytes
    local_segment_size = args.local_segment_size * 1024 * 1024   # Convert MB to bytes
    
    retcode = store.setup(
        local_hostname=args.local_hostname,
        metadata_server=args.metadata_server,
        global_segment_size=global_segment_size,
        local_buffer_size=local_segment_size,
        protocol=args.protocol,
        device_name=args.device_name,
        master_server_address=args.master_server
    )
    
    if retcode != 0:
        raise RuntimeError(f"Failed to setup store client. Return code: {retcode}")
    
    return store


async def process_single_request(
    store: MooncakeDistributedStore, 
    request: KVCacheRequest
) -> RequestResult:
    """Process a single KV-cache request."""
    result = RequestResult(request_id=request.request_id)
    start_time = time.perf_counter()
    
    try:
        # Phase 1: Check existence of blocks (batch operation)
        exist_start = time.perf_counter()
        exist_results = store.batch_is_exist(request.block_hashes)
        result.exist_check_time = time.perf_counter() - exist_start
        
        # Count hits/misses
        cache_hits = sum(exist_results)
        cache_misses = len(exist_results) - cache_hits
        result.cache_hits = cache_hits
        result.cache_misses = cache_misses
        
        # Phase 2: Get data for existing blocks
        get_start = time.perf_counter()
        bytes_retrieved = 0
        
        # Find which blocks exist and get them
        existing_hashes = []
        for i, exists in enumerate(exist_results):
            if exists:
                existing_hashes.append(request.block_hashes[i])
        
        if existing_hashes:
            # Batch get operation
            data_results = store.batch_get_buffer(existing_hashes)
            for buffer in data_results:
                if buffer is not None:
                    bytes_retrieved += buffer.size()
        
        result.get_data_time = time.perf_counter() - get_start
        result.bytes_retrieved = bytes_retrieved
        result.success = True
        
    except Exception as e:
        result.error = str(e)
        result.success = False
    
    result.latency = time.perf_counter() - start_time
    return result


async def run_benchmark(
    args: argparse.Namespace,
    store: MooncakeDistributedStore,
    requests: List[KVCacheRequest]
) -> Tuple[BenchmarkMetrics, List[RequestResult]]:
    """Run the main benchmark."""
    print(f"Starting benchmark with {len(requests)} requests...")
    
    # Limit concurrency if specified
    semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
    
    async def limited_process_request(request):
        if semaphore is None:
            return await process_single_request(store, request)
        async with semaphore:
            return await process_single_request(store, request)
    
    # Create progress bar
    pbar = None if args.disable_tqdm else tqdm(total=len(requests))
    
    # Process requests
    benchmark_start = time.perf_counter()
    
    if args.request_rate == float("inf"):
        # Send all requests immediately
        tasks = []
        for request in requests:
            task = asyncio.create_task(limited_process_request(request))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    else:
        # Rate-limited sending
        results = []
        for i, request in enumerate(requests):
            if i > 0:
                # Wait according to request rate
                interval = np.random.exponential(1.0 / args.request_rate)
                await asyncio.sleep(interval)
            
            result = await limited_process_request(request)
            results.append(result)
            
            if pbar:
                pbar.update(1)
    
    benchmark_duration = time.perf_counter() - benchmark_start
    
    if pbar:
        pbar.close()
    
    # Calculate metrics
    metrics = calculate_metrics(requests, results, benchmark_duration)
    
    return metrics, results


def calculate_metrics(
    requests: List[KVCacheRequest],
    results: List[RequestResult], 
    duration: float
) -> BenchmarkMetrics:
    """Calculate benchmark metrics."""
    
    completed = sum(1 for r in results if r.success)
    total_blocks_checked = sum(len(req.block_hashes) for req in requests)
    total_cache_hits = sum(r.cache_hits for r in results if r.success)
    total_cache_misses = sum(r.cache_misses for r in results if r.success)
    total_bytes_retrieved = sum(r.bytes_retrieved for r in results if r.success)
    
    # Latency metrics
    successful_results = [r for r in results if r.success]
    latencies = [r.latency * 1000 for r in successful_results]  # Convert to ms
    exist_check_times = [r.exist_check_time * 1000 for r in successful_results]
    get_data_times = [r.get_data_time * 1000 for r in successful_results]
    
    return BenchmarkMetrics(
        completed=completed,
        total_requests=len(requests),
        total_blocks_checked=total_blocks_checked,
        total_cache_hits=total_cache_hits,
        total_cache_misses=total_cache_misses,
        total_bytes_retrieved=total_bytes_retrieved,
        request_throughput=completed / duration,
        block_throughput=total_blocks_checked / duration,
        cache_hit_rate=total_cache_hits / max(total_blocks_checked, 1),
        mean_latency_ms=np.mean(latencies) if latencies else 0,
        median_latency_ms=np.median(latencies) if latencies else 0,
        p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
        mean_exist_check_ms=np.mean(exist_check_times) if exist_check_times else 0,
        mean_get_data_ms=np.mean(get_data_times) if get_data_times else 0,
        duration=duration,
    )


def print_results(metrics: BenchmarkMetrics, args: argparse.Namespace):
    """Print benchmark results."""
    print("\n{s:{c}^{n}}".format(s=" KV-Cache Benchmark Results ", n=50, c="="))
    print("{:<40} {:<10}".format("Dataset:", args.dataset_name))
    print("{:<40} {:<10}".format("Request rate:", args.request_rate))
    print("{:<40} {:<10}".format("Max concurrency:", args.max_concurrency or "unlimited"))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", metrics.duration))
    print("{:<40} {:<10}".format("Total blocks checked:", metrics.total_blocks_checked))
    print("{:<40} {:<10}".format("Cache hits:", metrics.total_cache_hits))
    print("{:<40} {:<10}".format("Cache misses:", metrics.total_cache_misses))
    print("{:<40} {:<10.2%}".format("Cache hit rate:", metrics.cache_hit_rate))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Block throughput (blocks/s):", metrics.block_throughput))
    print("{:<40} {:<10.2f}".format("Data retrieved (MB):", metrics.total_bytes_retrieved / (1024**2)))
    print("{s:{c}^{n}}".format(s="Latency Metrics", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean latency (ms):", metrics.mean_latency_ms))
    print("{:<40} {:<10.2f}".format("Median latency (ms):", metrics.median_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 latency (ms):", metrics.p99_latency_ms))
    print("{:<40} {:<10.2f}".format("Mean exist check (ms):", metrics.mean_exist_check_ms))
    print("{:<40} {:<10.2f}".format("Mean get data (ms):", metrics.mean_get_data_ms))
    print("=" * 50)


def save_results(metrics: BenchmarkMetrics, results: List[RequestResult], args: argparse.Namespace):
    """Save results to JSON file."""
    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        total_requests = metrics.total_requests
        output_file = f"mooncake_kv_cache_{timestamp}_{total_requests}.json"
    
    result_data = {
        # Configuration
        "dataset_name": args.dataset_name,
        "num_requests": metrics.total_requests,
        "request_rate": args.request_rate,
        "max_concurrency": args.max_concurrency,
        "block_size_tokens": args.block_size_tokens,
        "token_size_kb": args.token_size_kb,
        "global_segment_size_mb": args.global_segment_size,
        "local_segment_size_mb": args.local_segment_size,
        
        # Results
        "duration": metrics.duration,
        "completed": metrics.completed,
        "cache_hit_rate": metrics.cache_hit_rate,
        "request_throughput": metrics.request_throughput,
        "block_throughput": metrics.block_throughput,
        "mean_latency_ms": metrics.mean_latency_ms,
        "median_latency_ms": metrics.median_latency_ms,
        "p99_latency_ms": metrics.p99_latency_ms,
        "mean_exist_check_ms": metrics.mean_exist_check_ms,
        "mean_get_data_ms": metrics.mean_get_data_ms,
        "total_bytes_retrieved": metrics.total_bytes_retrieved,
    }
    
    if args.output_details:
        result_data["request_details"] = [
            {
                "request_id": r.request_id,
                "success": r.success,
                "latency_ms": r.latency * 1000,
                "cache_hits": r.cache_hits,
                "cache_misses": r.cache_misses,
                "bytes_retrieved": r.bytes_retrieved,
                "error": r.error,
            }
            for r in results
        ]
    
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark MoonCake Store with KV-cache workload simulation"
    )
    
    # Store configuration
    parser.add_argument(
        "--master-server", type=str, default="127.0.0.1:50051",
        help="Master server address"
    )
    parser.add_argument(
        "--metadata-server", type=str, default="127.0.0.1:2379",
        help="Metadata server address" 
    )
    parser.add_argument(
        "--local-hostname", type=str, default="localhost",
        help="Local hostname"
    )
    parser.add_argument(
        "--protocol", type=str, default="tcp", choices=["tcp", "rdma"],
        help="Communication protocol"
    )
    parser.add_argument(
        "--device-name", type=str, default="",
        help="Device name for RDMA"
    )
    parser.add_argument(
        "--global-segment-size", type=int, default=3200,
        help="Global segment size in MB"
    )
    parser.add_argument(
        "--local-segment-size", type=int, default=512,
        help="Local segment size in MB"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset-name", type=str, default="generated-shared-prefix",
        choices=["generated-shared-prefix"],
        help="Dataset type to use"
    )
    parser.add_argument(
        "--block-size-tokens", type=int, default=256,
        help="Number of tokens per block"
    )
    parser.add_argument(
        "--token-size-kb", type=int, default=160,
        help="Size of each token in KB (e.g., 160KB for Qwen3-32B)"
    )
    
    # Generated shared prefix configuration
    parser.add_argument(
        "--gsp-num-groups", type=int, default=64,
        help="Number of system prompt groups"
    )
    parser.add_argument(
        "--gsp-prompts-per-group", type=int, default=16,
        help="Number of prompts per group"
    )
    parser.add_argument(
        "--gsp-system-prompt-len", type=int, default=2048,
        help="System prompt length in tokens"
    )
    parser.add_argument(
        "--gsp-question-len", type=int, default=128,
        help="Question length in tokens"
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--request-rate", type=float, default=float("inf"),
        help="Request rate (requests per second). Inf means send all at once"
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=None,
        help="Maximum concurrent requests"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-file", type=str, help="Output JSON file path"
    )
    parser.add_argument(
        "--output-details", action="store_true",
        help="Include detailed per-request results in output"
    )
    parser.add_argument(
        "--disable-tqdm", action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


async def main():
    """Main benchmark function."""
    args = parse_arguments()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Calculate total requests
    total_requests = args.gsp_num_groups * args.gsp_prompts_per_group
    
    print(f"KV-Cache Benchmark Configuration:")
    print(f"Master server: {args.master_server}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Groups: {args.gsp_num_groups} × {args.gsp_prompts_per_group} = {total_requests} requests")
    print(f"Block size: {args.block_size_tokens} tokens ({args.token_size_kb} KB)")
    print(f"Store: {args.global_segment_size}MB global, {args.local_segment_size}MB local")
    print()
    
    try:
        # Setup store
        print("Setting up MoonCake Store client...")
        store = setup_store_client(args)
        print("Store client ready\n")
        
        # Generate requests
        if args.dataset_name == "generated-shared-prefix":
            requests = sample_generated_shared_prefix_requests(
                num_groups=args.gsp_num_groups,
                prompts_per_group=args.gsp_prompts_per_group,
                system_prompt_len=args.gsp_system_prompt_len,
                question_len=args.gsp_question_len,
                block_size_tokens=args.block_size_tokens,
                token_size_kb=args.token_size_kb,
                args=args,
            )
        else:
            raise ValueError(f"Unknown dataset: {args.dataset_name}")
        
        # Run benchmark
        metrics, results = await run_benchmark(args, store, requests)
        
        # Print and save results
        print_results(metrics, args)
        save_results(metrics, results, args)
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

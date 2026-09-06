#!/usr/bin/env python3
"""Small throughput benchmark for the local vLLM service.

The goal is not to replace vLLM's official benchmark suite. This script gives
task-level evidence for the report: serial requests versus concurrent requests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI


DEFAULT_PROMPT = (
    "You are an ALFWorld agent. Think briefly, then output one valid text action. "
    "Observation: You are in the middle of a room. You see a fridge 1, countertop 1, "
    "and microwave 1. Your task is to: heat some apple and put it on countertop 1.\n>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--trust-env",
        action="store_true",
        help="Honor proxy environment variables. Disabled by default for local 127.0.0.1 serving.",
    )
    parser.add_argument("--output", default="logs/benchmark_task1_generation.json")
    parser.add_argument(
        "--stop-newline",
        action="store_true",
        help="Stop at the first newline, matching the later ReAct one-action call pattern.",
    )
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, round((len(values) - 1) * p))
    return values[index]


def completion_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage and getattr(usage, "completion_tokens", None) is not None:
        return int(usage.completion_tokens)
    text = response.choices[0].text
    return max(1, len(text) // 4)


def one_request(args: argparse.Namespace, request_id: int) -> dict[str, Any]:
    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
        http_client=httpx.Client(timeout=args.timeout, trust_env=args.trust_env),
    )
    started = time.perf_counter()
    response = client.completions.create(
        model=args.model,
        prompt=f"{args.prompt}\nRequest id: {request_id}\n>",
        temperature=0,
        max_tokens=args.max_tokens,
        stop=["\n"] if args.stop_newline else None,
    )
    elapsed = time.perf_counter() - started
    text = response.choices[0].text.strip()
    return {
        "request_id": request_id,
        "latency_sec": elapsed,
        "completion_tokens": completion_tokens(response),
        "text": text,
    }


def run_setting(args: argparse.Namespace, concurrency: int) -> dict[str, Any]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(one_request, args, i) for i in range(args.num_requests)]
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]
    elapsed = time.perf_counter() - started

    latencies = [row["latency_sec"] for row in rows]
    total_tokens = sum(row["completion_tokens"] for row in rows)
    return {
        "concurrency": concurrency,
        "num_requests": args.num_requests,
        "wall_time_sec": elapsed,
        "completion_tokens": total_tokens,
        "completion_tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        "latency_avg_sec": statistics.mean(latencies),
        "latency_p50_sec": percentile(latencies, 0.50),
        "latency_p95_sec": percentile(latencies, 0.95),
        "samples": rows[:3],
    }


def main() -> None:
    args = parse_args()
    results = [run_setting(args, concurrency) for concurrency in args.concurrency]
    payload = {
        "base_url": args.base_url,
        "model": args.model,
        "num_requests": args.num_requests,
        "max_tokens": args.max_tokens,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


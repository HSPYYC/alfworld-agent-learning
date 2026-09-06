#!/usr/bin/env python3
"""Smoke test for the local OpenAI-compatible Qwen service."""

from __future__ import annotations

import argparse

import httpx
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--prompt", default="你好，请用一句话说明你已经可以被本地脚本调用。")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--api",
        choices=["completions", "chat"],
        default="completions",
        help="Use completions for the original ReAct prompt style; chat is only a sanity check.",
    )
    parser.add_argument(
        "--trust-env",
        action="store_true",
        help="Honor proxy environment variables. Disabled by default for local 127.0.0.1 serving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
        http_client=httpx.Client(trust_env=args.trust_env),
    )

    if args.api == "chat":
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            temperature=0,
            max_tokens=args.max_tokens,
        )
        text = response.choices[0].message.content or ""
    else:
        response = client.completions.create(
            model=args.model,
            prompt=args.prompt,
            temperature=0,
            max_tokens=args.max_tokens,
        )
        text = response.choices[0].text

    print(text.strip())


if __name__ == "__main__":
    main()


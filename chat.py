#!/usr/bin/env python3
"""
Multi-turn chat client for vLLM with per-query performance metrics.

Usage:
    python3 chat.py                          # defaults: localhost:8000, Qwen2.5-1.5B-Instruct
    python3 chat.py --base-url http://host:8000/v1 --model Qwen/Qwen2.5-VL-7B-Instruct
"""

import argparse
import subprocess
import sys
import time
from collections import deque

from openai import OpenAI


# ── GPU Memory ──────────────────────────────────────────────────

def get_gpu_memory_mb():
    """Get GPU memory used (MB) via nvidia-smi. Returns None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,nounits,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split("\n")[0]
        used, total = [int(x.strip()) for x in out.split(",")]
        return used, total
    except Exception:
        return None, None


# ── Metrics Display ─────────────────────────────────────────────

def print_metrics(ttft_ms, total_s, prompt_tok, completion_tok, gpu_used, gpu_total):
    tps = completion_tok / total_s if total_s > 0 and completion_tok > 0 else 0
    print()
    print("+" + "-" * 50 + "+")
    print("|  PERFORMANCE METRICS" + " " * 29 + "|")
    print("+" + "-" * 50 + "+")
    if ttft_ms is not None:
        print(f"|  TTFT:              {ttft_ms:>8.1f} ms" + " " * 16 + "|")
    print(f"|  Total time:        {total_s:>8.3f} s" + " " * 17 + "|")
    print(f"|  Prompt tokens:     {prompt_tok:>8d}" + " " * 19 + "|")
    print(f"|  Completion tokens: {completion_tok:>8d}" + " " * 19 + "|")
    print(f"|  Total tokens:      {prompt_tok + completion_tok:>8d}" + " " * 19 + "|")
    print(f"|  Generation speed:  {tps:>8.1f} tokens/s" + " " * 10 + "|")
    if gpu_used is not None:
        print(f"|  GPU memory:     {gpu_used:>5d} / {gpu_total:>5d} MB" + " " * 11 + "|")
    print("+" + "-" * 50 + "+")
    print()


# ── Chat Client ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="vLLM multi-turn chat with metrics")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vLLM server URL (default: http://localhost:8000/v1)")
    parser.add_argument("--model", default="Qwen/Qwen3-32B",
                        help="Model name")
    parser.add_argument("--max-history", type=int, default=20,
                        help="Max conversation turns to keep (default: 20)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per response (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--system-prompt", default="You are a helpful, concise AI assistant.",
                        help="System prompt")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    history = deque(maxlen=args.max_history)

    print("=" * 52)
    print("  vLLM Multi-Turn Chat")
    print(f"  Model: {args.model}")
    print(f"  Server: {args.base_url}")
    print("=" * 52)
    print("  Commands:  /quit  /reset  /history")
    print("=" * 52)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            history.clear()
            print("Conversation history cleared.\n")
            continue

        if user_input.lower() == "/history":
            if not history:
                print("(empty)\n")
                continue
            print()
            for i, msg in enumerate(history, 1):
                role = msg["role"].upper()
                text = msg["content"]
                if len(text) > 80:
                    text = text[:80] + "..."
                print(f"  {i}. [{role}] {text}")
            print()
            continue

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        # Build messages
        messages = [{"role": "system", "content": args.system_prompt}]
        messages.extend(history)

        # --- Streaming call (for TTFT + live output) ---
        t_start = time.perf_counter()
        ttft_ms = None
        response_text = ""
        chunk_count = 0
        usage_info = None

        try:
            print("\nAssistant: ", end="", flush=True)
            stream = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )

            for chunk in stream:
                # The final chunk may have usage info
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_info = chunk.usage
                    continue

                if not chunk.choices:
                    continue

                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000

                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    response_text += delta
                    chunk_count += 1

            print()  # newline after streamed output
            t_total = time.perf_counter() - t_start

        except Exception as e:
            print(f"\nError: {e}\n")
            history.pop()  # remove the failed user message
            continue

        # Add assistant response to history
        history.append({"role": "assistant", "content": response_text})

        # Token counts: prefer API usage info, fall back to chunk count
        if usage_info:
            prompt_tokens = usage_info.prompt_tokens
            completion_tokens = usage_info.completion_tokens
        else:
            prompt_tokens = 0
            completion_tokens = chunk_count  # one token per chunk is a reasonable estimate

        gpu_used, gpu_total = get_gpu_memory_mb()
        print_metrics(ttft_ms, t_total, prompt_tokens, completion_tokens, gpu_used, gpu_total)


if __name__ == "__main__":
    main()

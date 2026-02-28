#!/usr/bin/env python3
"""
LLM Inference Benchmark — sweeps across concurrent users and prompt sizes,
reporting percentile latency metrics and throughput.

Works with any OpenAI-compatible endpoint (vLLM, TACO-X, etc.)

Usage:
    python3 benchmark.py                                          # full sweep (localhost:18080)
    python3 benchmark.py --base-url http://1.2.3.4:18080/v1      # remote server
    python3 benchmark.py --concurrency 1,3 --prompt-sizes short --num-requests 5
    python3 benchmark.py --save results.csv
    python3 benchmark.py --save results.json                      # JSON output
"""

import argparse
import asyncio
import csv
import json
import subprocess
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from openai import AsyncOpenAI


# ── Fixed Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a helpful assistant."

PROMPTS = {
    "short": (
        "Explain cloud computing in simple terms. "
        "Cover what it is, why businesses use it, and give two real-world examples."
    ),
    "medium": (
        "Analyze the following product description and provide a competitive assessment.\n\n"
        "Product: CloudSync Pro — Enterprise File Synchronization Platform\n\n"
        "CloudSync Pro is an enterprise-grade file synchronization and sharing platform "
        "designed for organizations with 500+ employees. The platform offers real-time "
        "file synchronization across Windows, macOS, Linux, iOS, and Android devices. "
        "Key features include end-to-end AES-256 encryption, granular access controls "
        "with role-based permissions, integration with Active Directory and SAML 2.0 "
        "identity providers, and comprehensive audit logging for compliance.\n\n"
        "The platform supports files up to 50GB in size and offers delta sync technology "
        "that only transfers changed portions of files, reducing bandwidth usage by up to "
        "90%. CloudSync Pro includes a built-in document editor for basic collaboration, "
        "version history with unlimited retention, and automated backup scheduling. The "
        "admin console provides real-time dashboards showing storage utilization, active "
        "users, sync status, and security alerts.\n\n"
        "Pricing starts at $12 per user per month for the Business tier (minimum 100 users) "
        "and $20 per user per month for the Enterprise tier which adds advanced DLP policies, "
        "custom branding, dedicated support, and 99.99% uptime SLA. On-premises deployment "
        "is available for the Enterprise tier at custom pricing.\n\n"
        "Questions: 1) What are the three strongest competitive advantages? "
        "2) What gaps might enterprise buyers identify? "
        "3) How does the pricing compare to Dropbox Business and Box Enterprise?"
    ),
    "long": (
        "Summarize the following article and extract the five most important takeaways.\n\n"
        "Article: The Evolution of Distributed Systems Architecture\n\n"
        "The history of distributed systems architecture spans over five decades, from "
        "the earliest time-sharing systems of the 1960s to today's globally distributed "
        "cloud-native applications. Understanding this evolution is crucial for architects "
        "designing modern systems, as many fundamental challenges identified decades ago "
        "remain relevant today.\n\n"
        "In the 1970s, the emergence of ARPANET introduced the concept of networked "
        "computing, where multiple independent computers could communicate and share "
        "resources. Early distributed systems faced challenges with network reliability, "
        "consistency, and fault tolerance. The CAP theorem, though not formally described "
        "until 2000 by Eric Brewer, was already being grappled with in practice. Researchers "
        "at Xerox PARC developed early RPC mechanisms, laying the groundwork for inter-process "
        "communication patterns that would become fundamental to distributed computing.\n\n"
        "The 1980s and 1990s saw the rise of client-server architecture, which became the "
        "dominant paradigm for enterprise applications. Databases like Oracle and SQL Server "
        "centralized data management while distributed computing frameworks like CORBA and "
        "DCOM attempted to standardize remote object communication. These frameworks, while "
        "ambitious, suffered from complexity and tight coupling between components. The Java "
        "RMI and Enterprise JavaBeans era further attempted to simplify distributed development "
        "but introduced its own complexity through heavy middleware layers.\n\n"
        "The early 2000s brought a paradigm shift with Service-Oriented Architecture (SOA). "
        "SOA promoted loose coupling through well-defined service interfaces, typically using "
        "SOAP and XML for communication. While SOA addressed some of the coupling issues of "
        "earlier approaches, it often led to complex enterprise service bus (ESB) deployments "
        "that became bottlenecks themselves. The WS-* specification stack grew increasingly "
        "complex, leading many developers to seek simpler alternatives.\n\n"
        "The REST architectural style, described by Roy Fielding in his 2000 dissertation, "
        "gained widespread adoption in the mid-2000s as a simpler alternative to SOAP-based "
        "services. RESTful APIs leveraged HTTP semantics and JSON payloads, dramatically "
        "reducing the complexity of service interfaces. This simplification, combined with "
        "the rise of cloud computing platforms like AWS (launched 2006), enabled a new wave "
        "of distributed applications.\n\n"
        "The microservices architectural style emerged around 2011-2014, popularized by "
        "companies like Netflix, Amazon, and Spotify. Microservices decompose applications "
        "into small, independently deployable services, each owning its data and business "
        "logic. Key principles include single responsibility, independent deployability, "
        "decentralized data management, and design for failure. Netflix's engineering team "
        "published extensively about their experience, including the development of tools "
        "like Hystrix for circuit breaking, Eureka for service discovery, and Zuul for API "
        "gateway functionality.\n\n"
        "Container technology, particularly Docker (released 2013) and Kubernetes (released "
        "2014), provided the infrastructure foundation that made microservices practical at "
        "scale. Containers offered consistent deployment units, while Kubernetes provided "
        "orchestration capabilities including service discovery, load balancing, rolling "
        "updates, and self-healing. The combination of microservices architecture with "
        "container orchestration became the de facto standard for cloud-native applications.\n\n"
        "The serverless computing model, introduced with AWS Lambda in 2014, pushed "
        "abstraction further by eliminating server management entirely. Functions-as-a-Service "
        "(FaaS) allows developers to deploy individual functions that scale automatically "
        "and charge only for actual execution time. While serverless excels for event-driven "
        "workloads and variable traffic patterns, it introduces challenges around cold starts, "
        "vendor lock-in, debugging complexity, and state management.\n\n"
        "Event-driven architecture and event sourcing have gained renewed interest as "
        "organizations deal with increasingly complex data flows. Apache Kafka, originally "
        "developed at LinkedIn, has become the backbone of many event-driven systems, "
        "providing durable, high-throughput event streaming. Event sourcing, which stores "
        "state changes as an immutable sequence of events, combined with CQRS (Command Query "
        "Responsibility Segregation), offers powerful patterns for building systems that need "
        "complete audit trails and the ability to reconstruct state at any point in time.\n\n"
        "The service mesh pattern, implemented by tools like Istio and Linkerd, addresses "
        "cross-cutting concerns in microservices architectures. By deploying sidecar proxies "
        "alongside each service instance, service meshes provide transparent mTLS encryption, "
        "traffic management, observability, and resilience features without requiring changes "
        "to application code. This separation of concerns allows development teams to focus "
        "on business logic while platform teams manage infrastructure concerns.\n\n"
        "Looking forward, several trends are shaping the next generation of distributed "
        "systems. Edge computing pushes computation closer to data sources, reducing latency "
        "for IoT and real-time applications. WebAssembly (Wasm) is emerging as a portable, "
        "secure execution environment for distributed computing. AI-driven operations (AIOps) "
        "promise to automate the increasingly complex task of managing distributed systems. "
        "And new consistency models and distributed databases like CockroachDB and TiDB are "
        "making globally distributed transactions more practical.\n\n"
        "The fundamental challenges of distributed systems — network partitions, consistency, "
        "availability, latency, and observability — remain as relevant as ever. What has "
        "changed is the tooling, abstractions, and patterns available to address them. Modern "
        "architects must navigate an increasingly rich landscape of options, making informed "
        "trade-offs based on their specific requirements for consistency, availability, "
        "performance, and operational complexity."
    ),
}

MULTI_TURN_CONVERSATION = [
    "What are the main differences between TCP and UDP?",
    "Can you give me a real-world example where UDP is preferred over TCP?",
    "How does QUIC improve on both TCP and UDP?",
    "Summarize the trade-offs between TCP, UDP, and QUIC in a table format.",
    "Based on everything we discussed, what would you recommend for a video streaming application and why?",
]


# ── Data Structures ────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    ttft_ms: Optional[float] = None
    e2e_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None

    @property
    def tpot_ms(self) -> Optional[float]:
        """Time per output token: (E2E - TTFT) / (output_tokens - 1)."""
        if self.ttft_ms is None or self.completion_tokens <= 1:
            return None
        return ((self.e2e_s * 1000) - self.ttft_ms) / (self.completion_tokens - 1)


@dataclass
class ConfigResult:
    concurrency: int = 0
    prompt_size: str = ""
    max_tokens: int = 0
    results: list = field(default_factory=list)

    @property
    def successful(self) -> list:
        return [r for r in self.results if r.error is None]

    @property
    def failures(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    def percentile(self, values: list, p: int) -> Optional[float]:
        if not values:
            return None
        sorted_v = sorted(values)
        k = (len(sorted_v) - 1) * (p / 100)
        f = int(k)
        c = f + 1
        if c >= len(sorted_v):
            return sorted_v[f]
        return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])

    def ttft_percentiles(self) -> dict:
        vals = [r.ttft_ms for r in self.successful if r.ttft_ms is not None]
        return {f"p{p}": self.percentile(vals, p) for p in (50, 90, 99)}

    def tpot_percentiles(self) -> dict:
        vals = [r.tpot_ms for r in self.successful if r.tpot_ms is not None]
        return {f"p{p}": self.percentile(vals, p) for p in (50, 90, 99)}

    def e2e_percentiles(self) -> dict:
        vals = [r.e2e_s for r in self.successful]
        return {f"p{p}": self.percentile(vals, p) for p in (50, 90, 99)}

    def throughput_tok_s(self) -> float:
        """Aggregate output tokens / wall-clock time for the batch."""
        total_tokens = sum(r.completion_tokens for r in self.successful)
        total_time = sum(r.e2e_s for r in self.successful)
        if total_time == 0:
            return 0.0
        if not self.successful:
            return 0.0
        avg_e2e = total_time / len(self.successful)
        return total_tokens / avg_e2e if avg_e2e > 0 else 0.0


# ── Core Benchmark Logic ──────────────────────────────────────────────────

async def send_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    messages: Optional[list] = None,
) -> RequestResult:
    """Send a single streaming chat request and collect metrics."""
    if messages is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    result = RequestResult()
    t_start = time.perf_counter()

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                result.prompt_tokens = chunk.usage.prompt_tokens
                result.completion_tokens = chunk.usage.completion_tokens
                continue
            if not chunk.choices:
                continue
            if result.ttft_ms is None:
                result.ttft_ms = (time.perf_counter() - t_start) * 1000

        result.e2e_s = time.perf_counter() - t_start

    except Exception as e:
        result.e2e_s = time.perf_counter() - t_start
        result.error = str(e)

    return result


async def run_config(
    client: AsyncOpenAI,
    model: str,
    concurrency: int,
    prompt_size: str,
    max_tokens: int,
    num_requests: int,
    warmup: int,
) -> ConfigResult:
    """Run a single benchmark configuration: warmup then timed requests."""
    prompt = PROMPTS[prompt_size]
    sem = asyncio.Semaphore(concurrency)

    async def limited_request():
        async with sem:
            return await send_request(client, model, prompt, max_tokens)

    # Warmup phase (not timed)
    if warmup > 0:
        warmup_tasks = [limited_request() for _ in range(warmup)]
        await asyncio.gather(*warmup_tasks)

    # Timed phase
    timed_tasks = [limited_request() for _ in range(num_requests)]
    results = await asyncio.gather(*timed_tasks)

    config = ConfigResult(
        concurrency=concurrency,
        prompt_size=prompt_size,
        max_tokens=max_tokens,
        results=list(results),
    )
    return config


async def run_multi_turn(
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 256,
) -> list:
    """Run a multi-turn conversation, measuring per-turn metrics."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn_results = []

    for turn_idx, user_msg in enumerate(MULTI_TURN_CONVERSATION, 1):
        messages.append({"role": "user", "content": user_msg})

        t_start = time.perf_counter()
        ttft_ms = None
        response_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    continue
                if not chunk.choices:
                    continue
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                delta = chunk.choices[0].delta.content
                if delta:
                    response_text += delta

            e2e_s = time.perf_counter() - t_start
            tpot_ms = None
            if ttft_ms is not None and completion_tokens > 1:
                tpot_ms = ((e2e_s * 1000) - ttft_ms) / (completion_tokens - 1)
            tok_s = completion_tokens / e2e_s if e2e_s > 0 else 0

            messages.append({"role": "assistant", "content": response_text})

            turn_results.append({
                "turn": turn_idx,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "e2e_s": e2e_s,
                "tok_s": tok_s,
                "error": None,
            })

        except Exception as e:
            turn_results.append({
                "turn": turn_idx,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "ttft_ms": None,
                "tpot_ms": None,
                "e2e_s": time.perf_counter() - t_start,
                "tok_s": 0,
                "error": str(e),
            })
            messages.append({"role": "assistant", "content": "(error)"})

    return turn_results


# ── Helpers ────────────────────────────────────────────────────────────────

def get_gpu_memory_mb():
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


def fmt(val, suffix="", decimals=1, width=8):
    """Format a numeric value, returning '—' if None."""
    if val is None:
        return "—".rjust(width)
    return f"{val:.{decimals}f}{suffix}".rjust(width)


# ── Output Formatting ─────────────────────────────────────────────────────

def print_sweep_results(configs: list, num_requests: int):
    """Print the full sweep results table grouped by prompt size."""
    total_failures = sum(c.failures for c in configs)

    print()
    print("=" * 90)
    print("  SINGLE-REQUEST SWEEP RESULTS")
    print("=" * 90)

    # Legend
    print()
    print("  Metric Definitions:")
    print("    TTFT     Time To First Token — delay before model starts responding (ms)")
    print("    TPOT     Time Per Output Token — avg time between each generated token (ms)")
    print("    E2E      End-to-End latency — total time from request to last token (s)")
    print("    Tok/s    Throughput — output tokens generated per second")
    print("    p50      Median (50th percentile) — typical request performance")
    print("    p90      90th percentile — 1 in 10 requests is slower than this")
    print("    p99      99th percentile — worst-case tail latency (1 in 100)")
    print("    Conc     Concurrent users — simultaneous requests in flight")
    print()
    print(f"  {num_requests} requests per configuration")
    if total_failures > 0:
        print(f"  WARNING: {total_failures} total failed requests")
    print()

    header = (
        f"{'Prompt':<8} {'Conc':>4} │"
        f"{'TTFT p50':>9} {'p90':>8} {'p99':>8} │"
        f"{'TPOT p50':>9} {'p90':>8} │"
        f"{'E2E p50':>8} {'p90':>8} │"
        f"{'Tok/s':>7}"
    )
    # Add Errors column only if there are failures
    if total_failures > 0:
        header += f" {'Err':>4}"

    print(header)
    print("─" * 90)

    current_prompt = None
    for c in configs:
        if c.prompt_size != current_prompt:
            if current_prompt is not None:
                print("─" * 90)
            current_prompt = c.prompt_size

        ttft = c.ttft_percentiles()
        tpot = c.tpot_percentiles()
        e2e = c.e2e_percentiles()
        throughput = c.throughput_tok_s()

        line = (
            f"{c.prompt_size:<8} {c.concurrency:>4} │"
            f"{fmt(ttft['p50'], 'ms', 0, 9)} {fmt(ttft['p90'], 'ms', 0, 8)} {fmt(ttft['p99'], 'ms', 0, 8)} │"
            f"{fmt(tpot['p50'], 'ms', 1, 9)} {fmt(tpot['p90'], 'ms', 1, 8)} │"
            f"{fmt(e2e['p50'], 's', 2, 8)} {fmt(e2e['p90'], 's', 2, 8)} │"
            f"{throughput:>7.1f}"
        )
        if total_failures > 0:
            line += f" {c.failures:>4}"

        print(line)

    print("─" * 90)


def print_multi_turn_results(turn_results: list):
    """Print multi-turn conversation results."""
    print()
    print("=" * 90)
    print("  MULTI-TURN CONVERSATION (prefix caching test)")
    print("=" * 90)

    header = (
        f"{'Turn':>4} {'Prompt Tok':>10} {'Compl Tok':>9} │"
        f"{'TTFT':>9} {'TPOT':>9} {'E2E':>8} {'Tok/s':>7} {'Status':>7}"
    )
    print(header)
    print("─" * 70)

    for t in turn_results:
        status = "OK" if t["error"] is None else "FAIL"
        print(
            f"{t['turn']:>4} {t['prompt_tokens']:>10} {t['completion_tokens']:>9} │"
            f"{fmt(t['ttft_ms'], 'ms', 0, 9)} {fmt(t['tpot_ms'], 'ms', 1, 9)}"
            f"{fmt(t['e2e_s'], 's', 2, 8)} {t['tok_s']:>7.1f} {status:>7}"
        )

    print("─" * 70)

    # Highlight caching effect
    if len(turn_results) >= 2:
        t1_ttft = turn_results[0].get("ttft_ms")
        later_ttfts = [t["ttft_ms"] for t in turn_results[1:] if t["ttft_ms"] is not None]
        if t1_ttft and later_ttfts:
            avg_later = statistics.mean(later_ttfts)
            ratio = avg_later / t1_ttft if t1_ttft > 0 else 0
            print(f"\n  Turn 1 TTFT: {t1_ttft:.0f}ms │ Avg Turn 2-5 TTFT: {avg_later:.0f}ms │ Ratio: {ratio:.2f}x")
            if ratio < 1:
                print("  → Prefix caching appears ACTIVE (later turns faster)")
            else:
                print("  → Prefix caching not observed (TTFT grows with context)")


def save_results(filepath: str, configs: list, multi_turn: list,
                 gpu_info: tuple = (None, None), num_requests: int = 10):
    """Save results to CSV or JSON based on file extension."""
    if filepath.endswith(".csv"):
        save_csv(filepath, configs, multi_turn, gpu_info, num_requests)
    else:
        save_json(filepath, configs, multi_turn, gpu_info, num_requests)
    print(f"\nResults saved to {filepath}")


def save_csv(filepath: str, configs: list, multi_turn: list,
             gpu_info: tuple = (None, None), num_requests: int = 10):
    """Save results as a flat CSV with one row per config."""
    total_failures = sum(c.failures for c in configs)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # Metadata row
        writer.writerow([f"# {num_requests} requests per config"])
        writer.writerow([])

        # Sweep header
        sweep_header = [
            "prompt_size", "concurrency",
            "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
            "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms",
            "e2e_p50_s", "e2e_p90_s", "e2e_p99_s",
            "throughput_tok_s",
        ]
        if total_failures > 0:
            sweep_header.append("errors")
        writer.writerow(sweep_header)

        for c in configs:
            ttft = c.ttft_percentiles()
            tpot = c.tpot_percentiles()
            e2e = c.e2e_percentiles()
            row = [
                c.prompt_size, c.concurrency,
                f"{ttft['p50']:.1f}" if ttft["p50"] is not None else "",
                f"{ttft['p90']:.1f}" if ttft["p90"] is not None else "",
                f"{ttft['p99']:.1f}" if ttft["p99"] is not None else "",
                f"{tpot['p50']:.1f}" if tpot["p50"] is not None else "",
                f"{tpot['p90']:.1f}" if tpot["p90"] is not None else "",
                f"{tpot['p99']:.1f}" if tpot["p99"] is not None else "",
                f"{e2e['p50']:.2f}" if e2e["p50"] is not None else "",
                f"{e2e['p90']:.2f}" if e2e["p90"] is not None else "",
                f"{e2e['p99']:.2f}" if e2e["p99"] is not None else "",
                f"{c.throughput_tok_s():.1f}",
            ]
            if total_failures > 0:
                row.append(c.failures)
            writer.writerow(row)

        # Blank row separator
        writer.writerow([])

        # Multi-turn results
        writer.writerow([
            "turn", "prompt_tokens", "completion_tokens",
            "ttft_ms", "tpot_ms", "e2e_s", "tok_s",
        ])
        for t in multi_turn:
            writer.writerow([
                t["turn"], t["prompt_tokens"], t["completion_tokens"],
                f"{t['ttft_ms']:.1f}" if t["ttft_ms"] is not None else "",
                f"{t['tpot_ms']:.1f}" if t["tpot_ms"] is not None else "",
                f"{t['e2e_s']:.2f}", f"{t['tok_s']:.1f}",
            ])

        # GPU info (single snapshot)
        writer.writerow([])
        writer.writerow(["gpu_used_mb", "gpu_total_mb"])
        writer.writerow([gpu_info[0] or "", gpu_info[1] or ""])


def save_json(filepath: str, configs: list, multi_turn: list,
              gpu_info: tuple = (None, None), num_requests: int = 10):
    """Save all results to a JSON file."""
    data = {
        "num_requests_per_config": num_requests,
        "gpu_memory_mb": {"used": gpu_info[0], "total": gpu_info[1]},
        "sweep": [],
        "multi_turn": multi_turn,
    }

    for c in configs:
        entry = {
            "concurrency": c.concurrency,
            "prompt_size": c.prompt_size,
            "ttft": c.ttft_percentiles(),
            "tpot": c.tpot_percentiles(),
            "e2e": c.e2e_percentiles(),
            "throughput_tok_s": c.throughput_tok_s(),
        }
        if c.failures > 0:
            entry["errors"] = c.failures
        data["sweep"].append(entry)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Main ───────────────────────────────────────────────────────────────────

async def async_main():
    parser = argparse.ArgumentParser(
        description="vLLM Benchmark — sweep concurrency × prompt size"
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--concurrency", default="1,4,8,16,32",
                        help="Comma-separated concurrency levels (default: 1,4,8,16,32)")
    parser.add_argument("--prompt-sizes", default="short,medium,long",
                        help="Comma-separated prompt sizes (default: short,medium,long)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max completion tokens (default: 256)")
    parser.add_argument("--num-requests", type=int, default=10,
                        help="Requests per configuration (default: 10)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup requests before timing (default: 2)")
    parser.add_argument("--save", nargs="?", const="auto", metavar="FILE",
                        help="Save results (default: results-YYYYMMDD-HHMMSS.csv)")
    args = parser.parse_args()

    # Auto-generate filename with timestamp if --save used without a value
    if args.save == "auto":
        args.save = time.strftime("results-%Y%m%d-%H%M%S.csv")

    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    prompt_sizes = [x.strip() for x in args.prompt_sizes.split(",")]

    total_configs = len(concurrency_levels) * len(prompt_sizes)
    total_requests = total_configs * (args.num_requests + args.warmup)

    print("=" * 90)
    print("  LLM Inference Benchmark")
    print(f"  Model:       {args.model}")
    print(f"  Server:      {args.base_url}")
    print(f"  Concurrency: {concurrency_levels}")
    print(f"  Prompts:     {prompt_sizes}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Requests:    {args.num_requests}/config + {args.warmup} warmup  ({total_requests} total)")
    print(f"  Configs:     {total_configs}")
    print("=" * 90)
    print()

    client = AsyncOpenAI(base_url=args.base_url, api_key="not-needed")

    # ── Single-request sweep ──
    all_configs = []
    config_num = 0

    for prompt_size in prompt_sizes:
        for conc in concurrency_levels:
            config_num += 1
            label = f"[{config_num}/{total_configs}] prompt={prompt_size} conc={conc}"
            print(f"  {label} ... ", end="", flush=True)

            config = await run_config(
                client, args.model, conc, prompt_size, args.max_tokens,
                args.num_requests, args.warmup,
            )
            all_configs.append(config)

            # One-line progress summary
            ttft_p50 = config.ttft_percentiles()["p50"]
            tpot_p50 = config.tpot_percentiles()["p50"]
            throughput = config.throughput_tok_s()
            ok = len(config.successful)
            fail = config.failures

            ttft_str = f"{ttft_p50:.0f}ms" if ttft_p50 is not None else "—"
            tpot_str = f"{tpot_p50:.1f}ms" if tpot_p50 is not None else "—"
            progress = f"TTFT_p50={ttft_str}  TPOT_p50={tpot_str}  tok/s={throughput:.1f}"
            if fail > 0:
                progress += f"  errors={fail}"
            print(progress)

    print_sweep_results(all_configs, args.num_requests)

    # ── Multi-turn test ──
    print("\n  Running multi-turn conversation (5 turns) ...", flush=True)
    multi_turn_results = await run_multi_turn(client, args.model, args.max_tokens)
    print_multi_turn_results(multi_turn_results)

    # ── GPU memory (single snapshot) ──
    gpu_used, gpu_total = get_gpu_memory_mb()
    if gpu_used is not None:
        print(f"\n  GPU Memory: {gpu_used} MB / {gpu_total} MB ({gpu_used/gpu_total*100:.0f}%)")
    else:
        print("\n  GPU Memory: not available (nvidia-smi not found)")

    # ── Save results ──
    if args.save:
        save_results(args.save, all_configs, multi_turn_results,
                     (gpu_used, gpu_total), args.num_requests)

    print()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

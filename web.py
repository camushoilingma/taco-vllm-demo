#!/usr/bin/env python3
"""
Web UI for TACO-X vs vLLM side-by-side comparison — type a prompt,
see both engines stream in parallel with real-time metrics.

Usage:
    python3 web.py                                                          # defaults
    python3 web.py --taco-url http://localhost:18080/v1                     # TACO-X endpoint
    python3 web.py --vllm-url http://43.153.138.241:8000/v1                # vLLM endpoint
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI
import uvicorn


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

app = FastAPI()

# Engine clients
taco_client: AsyncOpenAI = None
taco_model: str = ""
vllm_client: AsyncOpenAI = None
vllm_model: str = ""
warmup_file: str = None


@app.on_event("startup")
async def run_warmup():
    """Send warmup prompts on startup to populate the lookahead cache (TACO-X only)."""
    all_prompts = []

    if warmup_file:
        import json as _json
        with open(warmup_file) as f:
            all_prompts.extend(_json.load(f))

    all_prompts.extend(PROMPTS.values())

    if not all_prompts:
        return

    passes = 3
    total = len(all_prompts) * passes
    print(f"  Warmup (TACO-X only): {len(all_prompts)} prompts x {passes} passes = {total} requests...")
    n = 0
    for rnd in range(1, passes + 1):
        for p in all_prompts:
            n += 1
            prompt = p if isinstance(p, str) else p.get("prompt", "")
            try:
                resp = await taco_client.chat.completions.create(
                    model=taco_model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    max_tokens=256, temperature=0,
                )
                toks = resp.usage.completion_tokens if resp.usage else 0
                print(f"    [pass {rnd}, {n}/{total}] {toks} tokens")
            except Exception as e:
                print(f"    [pass {rnd}, {n}/{total}] FAIL: {e}")
    print("  Warmup complete.\n")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/api/presets")
async def presets():
    return {k: v for k, v in PROMPTS.items()}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.0)
    engine = body.get("engine", "taco")

    if engine == "vllm":
        api_client = vllm_client
        api_model = vllm_model
    else:
        api_client = taco_client
        api_model = taco_model

    async def event_stream():
        t_start = time.perf_counter()
        ttft_ms = None
        completion_tokens = 0
        prompt_tokens = 0

        try:
            stream = await api_client.chat.completions.create(
                model=api_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
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

                now = time.perf_counter()

                if ttft_ms is None:
                    ttft_ms = (now - t_start) * 1000

                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

            e2e_s = time.perf_counter() - t_start

            tpot_avg = None
            if completion_tokens > 1 and ttft_ms is not None:
                tpot_avg = ((e2e_s * 1000) - ttft_ms) / (completion_tokens - 1)

            tok_s = completion_tokens / e2e_s if e2e_s > 0 else 0

            metrics = {
                "type": "metrics",
                "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
                "tpot_ms": round(tpot_avg, 1) if tpot_avg else None,
                "e2e_s": round(e2e_s, 3),
                "tok_s": round(tok_s, 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            yield f"data: {json.dumps(metrics)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TACO-X vs vLLM</title>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #1c2128;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --accent-hover: #79c0ff;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --taco-accent: #f0883e;
    --vllm-accent: #58a6ff;
    --font-mono: 'SF Mono', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', Consolas, monospace;
    --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }

  header h1 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
  }

  header h1 .taco { color: var(--taco-accent); }
  header h1 .vllm { color: var(--vllm-accent); }

  header .server-info {
    font-size: 12px;
    color: var(--text-dim);
    font-family: var(--font-mono);
  }

  .input-section {
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .preset-buttons {
    display: flex;
    gap: 6px;
    padding: 10px 16px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }

  .preset-btn {
    padding: 5px 14px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface2);
    color: var(--text-dim);
    font-size: 12px;
    font-family: var(--font-mono);
    cursor: pointer;
    transition: all 0.15s;
  }

  .preset-btn:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(88, 166, 255, 0.08);
  }

  .preset-btn.active {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(88, 166, 255, 0.12);
  }

  .prompt-box {
    padding: 12px 16px;
    background: var(--bg);
  }

  textarea {
    width: 100%;
    min-height: 60px;
    max-height: 150px;
    padding: 10px 12px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.5;
    resize: vertical;
    outline: none;
    transition: border-color 0.15s;
  }

  textarea:focus { border-color: var(--accent); }
  textarea::placeholder { color: var(--text-dim); }

  .controls {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 16px 12px;
    background: var(--bg);
  }

  .param {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-dim);
  }

  .param input {
    width: 64px;
    padding: 4px 8px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 12px;
    outline: none;
  }

  .param input:focus { border-color: var(--accent); }

  .send-btn {
    margin-left: auto;
    padding: 7px 24px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }

  .send-btn:hover { background: var(--accent-hover); }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .send-btn.stop { background: var(--red); }

  /* Two-column comparison area */
  .compare-area {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .engine-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
  }

  .engine-col:first-child {
    border-right: 1px solid var(--border);
  }

  .engine-header {
    background: var(--surface);
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }

  .engine-header .engine-name {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .engine-header .taco-label { color: var(--taco-accent); }
  .engine-header .vllm-label { color: var(--vllm-accent); }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-dim);
    display: inline-block;
  }

  .status-dot.streaming { background: var(--green); animation: pulse 1.5s ease infinite; }
  .status-dot.done { background: var(--green); }
  .status-dot.error { background: var(--red); }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .engine-header .status-text {
    font-size: 11px;
    font-weight: 400;
    color: var(--text-dim);
    font-family: var(--font-mono);
  }

  .output-area {
    flex: 1;
    overflow-y: auto;
    padding: 14px 16px;
    font-family: var(--font-mono);
    font-size: 13px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--bg);
  }

  .output-area .error { color: var(--red); }

  .cursor-blink {
    display: inline-block;
    width: 8px;
    height: 16px;
    background: var(--accent);
    animation: blink 1s step-end infinite;
    vertical-align: text-bottom;
    margin-left: 1px;
  }

  @keyframes blink {
    50% { opacity: 0; }
  }

  .metrics-bar {
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 10px 16px;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    flex-shrink: 0;
  }

  .metric-item {
    text-align: center;
  }

  .metric-item .label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    margin-bottom: 2px;
  }

  .metric-item .value {
    font-size: 16px;
    font-weight: 700;
    font-family: var(--font-mono);
    color: var(--text);
  }

  .metric-item .value.highlight { color: var(--green); }

  .metric-item .unit {
    font-size: 10px;
    color: var(--text-dim);
    font-weight: 400;
  }

  .metric-item .value.winner { color: var(--green); }
  .metric-item .value.loser { color: var(--text-dim); }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-dim);
    font-size: 13px;
    gap: 8px;
  }

  .empty-state .icon {
    font-size: 32px;
    opacity: 0.3;
  }

</style>
</head>
<body>

<header>
  <h1><span class="taco">TACO-X</span> vs <span class="vllm">vLLM</span> &mdash; Side-by-Side Comparison</h1>
  <span class="server-info">Qwen3-32B FP16 &middot; 4x L20</span>
</header>

<div class="input-section">
  <div class="preset-buttons">
    <button class="preset-btn" onclick="loadPreset('short')">Short (~35 tok)</button>
    <button class="preset-btn" onclick="loadPreset('medium')">Medium (~290 tok)</button>
    <button class="preset-btn" onclick="loadPreset('long')">Long (~1000 tok)</button>
    <button class="preset-btn" onclick="clearPrompt()" style="margin-left:auto; color:var(--text-dim);">Clear</button>
  </div>
  <div class="prompt-box">
    <textarea id="promptInput" placeholder="Type your prompt here, or click a preset above..." rows="2"></textarea>
  </div>
  <div class="controls">
    <div class="param">
      <label>Max tokens</label>
      <input type="number" id="maxTokens" value="256" min="1" max="4096">
    </div>
    <div class="param">
      <label>Temp</label>
      <input type="number" id="temperature" value="0" min="0" max="2" step="0.1">
    </div>
    <button class="send-btn" id="sendBtn" onclick="sendPrompt()">Send</button>
  </div>
</div>

<div class="compare-area">
  <!-- TACO-X Column -->
  <div class="engine-col" id="tacoCol">
    <div class="engine-header">
      <div class="engine-name">
        <span class="status-dot" id="tacoDot"></span>
        <span class="taco-label">TACO-X</span>
        <span style="font-size:11px; font-weight:400; color:var(--text-dim);">(opt-level 3)</span>
      </div>
      <span class="status-text" id="tacoStatus">Ready</span>
    </div>
    <div class="output-area" id="tacoOutput">
      <div class="empty-state">
        <div class="icon">&#9655;</div>
        <div>Send a prompt to compare engines</div>
      </div>
    </div>
    <div class="metrics-bar" id="tacoMetrics">
      <div class="metric-item"><div class="label">TTFT</div><div class="value" id="tacoTTFT">--<span class="unit">ms</span></div></div>
      <div class="metric-item"><div class="label">TPOT</div><div class="value" id="tacoTPOT">--<span class="unit">ms</span></div></div>
      <div class="metric-item"><div class="label">E2E</div><div class="value" id="tacoE2E">--<span class="unit">s</span></div></div>
      <div class="metric-item"><div class="label">Tok/s</div><div class="value highlight" id="tacoTokS">--</div></div>
      <div class="metric-item"><div class="label">Tokens</div><div class="value" id="tacoTokens">--</div></div>
    </div>
  </div>

  <!-- vLLM Column -->
  <div class="engine-col" id="vllmCol">
    <div class="engine-header">
      <div class="engine-name">
        <span class="status-dot" id="vllmDot"></span>
        <span class="vllm-label">vLLM</span>
        <span style="font-size:11px; font-weight:400; color:var(--text-dim);">(CUDA graphs)</span>
      </div>
      <span class="status-text" id="vllmStatus">Ready</span>
    </div>
    <div class="output-area" id="vllmOutput">
      <div class="empty-state">
        <div class="icon">&#9655;</div>
        <div>Send a prompt to compare engines</div>
      </div>
    </div>
    <div class="metrics-bar" id="vllmMetrics">
      <div class="metric-item"><div class="label">TTFT</div><div class="value" id="vllmTTFT">--<span class="unit">ms</span></div></div>
      <div class="metric-item"><div class="label">TPOT</div><div class="value" id="vllmTPOT">--<span class="unit">ms</span></div></div>
      <div class="metric-item"><div class="label">E2E</div><div class="value" id="vllmE2E">--<span class="unit">s</span></div></div>
      <div class="metric-item"><div class="label">Tok/s</div><div class="value highlight" id="vllmTokS">--</div></div>
      <div class="metric-item"><div class="label">Tokens</div><div class="value" id="vllmTokens">--</div></div>
    </div>
  </div>
</div>

<script>
const presets = {};
let streaming = false;
let abortControllers = [];

async function init() {
  const resp = await fetch('/api/presets');
  const data = await resp.json();
  Object.assign(presets, data);
}

function loadPreset(name) {
  document.getElementById('promptInput').value = presets[name] || '';
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('promptInput').focus();
}

function clearPrompt() {
  document.getElementById('promptInput').value = '';
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('promptInput').focus();
}

function setMetric(id, value) {
  const el = document.getElementById(id);
  const unit = el.querySelector('.unit');
  el.textContent = value;
  if (unit) el.appendChild(unit);
}

function setStatus(engine, state, text) {
  const dot = document.getElementById(engine + 'Dot');
  dot.className = 'status-dot ' + state;
  document.getElementById(engine + 'Status').textContent = text;
}

function resetEngine(engine) {
  ['TTFT','TPOT','E2E','TokS','Tokens'].forEach(m => {
    setMetric(engine + m, '--');
  });
  setStatus(engine, '', 'Ready');
}

// Compare metrics and highlight winners after both complete
let engineMetrics = { taco: null, vllm: null };

function highlightWinners() {
  const t = engineMetrics.taco;
  const v = engineMetrics.vllm;
  if (!t || !v) return;

  // Lower is better: TTFT, TPOT, E2E
  [['TTFT', 'ttft_ms'], ['TPOT', 'tpot_ms'], ['E2E', 'e2e_s']].forEach(([label, key]) => {
    const tacoEl = document.getElementById('taco' + label);
    const vllmEl = document.getElementById('vllm' + label);
    tacoEl.classList.remove('winner', 'loser');
    vllmEl.classList.remove('winner', 'loser');
    if (t[key] != null && v[key] != null) {
      if (t[key] < v[key]) {
        tacoEl.classList.add('winner');
        vllmEl.classList.add('loser');
      } else if (v[key] < t[key]) {
        vllmEl.classList.add('winner');
        tacoEl.classList.add('loser');
      }
    }
  });

  // Higher is better: tok/s
  const tacoTokEl = document.getElementById('tacoTokS');
  const vllmTokEl = document.getElementById('vllmTokS');
  tacoTokEl.classList.remove('winner', 'loser');
  vllmTokEl.classList.remove('winner', 'loser');
  if (t.tok_s > v.tok_s) {
    tacoTokEl.classList.add('winner');
    vllmTokEl.classList.add('loser');
  } else if (v.tok_s > t.tok_s) {
    vllmTokEl.classList.add('winner');
    tacoTokEl.classList.add('loser');
  }
}

async function streamEngine(engine, prompt, maxTokens, temperature) {
  const output = document.getElementById(engine + 'Output');
  output.innerHTML = '';
  setStatus(engine, 'streaming', 'Streaming...');

  const cursor = document.createElement('span');
  cursor.className = 'cursor-blink';

  const ac = new AbortController();
  abortControllers.push(ac);

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt, max_tokens: maxTokens, temperature, engine }),
      signal: ac.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') continue;

        try {
          const msg = JSON.parse(payload);

          if (msg.type === 'token') {
            cursor.remove();
            output.appendChild(document.createTextNode(msg.content));
            output.appendChild(cursor);
            output.scrollTop = output.scrollHeight;
          }

          if (msg.type === 'metrics') {
            cursor.remove();
            if (msg.ttft_ms != null) setMetric(engine + 'TTFT', msg.ttft_ms);
            if (msg.tpot_ms != null) setMetric(engine + 'TPOT', msg.tpot_ms);
            setMetric(engine + 'E2E', msg.e2e_s);
            setMetric(engine + 'TokS', msg.tok_s);
            setMetric(engine + 'Tokens', msg.prompt_tokens + ' > ' + msg.completion_tokens);
            setStatus(engine, 'done', 'Done ' + msg.e2e_s + 's');

            engineMetrics[engine] = msg;
            highlightWinners();
          }

          if (msg.type === 'error') {
            cursor.remove();
            output.innerHTML += '<span class="error">\nError: ' + msg.content + '</span>';
            setStatus(engine, 'error', 'Error');
          }
        } catch(e) {}
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      setStatus(engine, 'done', 'Stopped');
    } else {
      setStatus(engine, 'error', 'Connection error');
      output.innerHTML += '<span class="error">\n' + e.message + '</span>';
    }
  }
}

async function sendPrompt() {
  const prompt = document.getElementById('promptInput').value.trim();
  if (!prompt) return;

  if (streaming) {
    abortControllers.forEach(ac => ac.abort());
    return;
  }

  const maxTokens = parseInt(document.getElementById('maxTokens').value) || 256;
  const temperature = parseFloat(document.getElementById('temperature').value) || 0;

  streaming = true;
  abortControllers = [];
  engineMetrics = { taco: null, vllm: null };

  const btn = document.getElementById('sendBtn');
  btn.textContent = 'Stop';
  btn.classList.add('stop');

  resetEngine('taco');
  resetEngine('vllm');

  // Fire both engines in parallel
  await Promise.allSettled([
    streamEngine('taco', prompt, maxTokens, temperature),
    streamEngine('vllm', prompt, maxTokens, temperature),
  ]);

  streaming = false;
  abortControllers = [];
  btn.textContent = 'Send';
  btn.classList.remove('stop');
}

document.getElementById('promptInput').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    sendPrompt();
  }
});

init();
</script>

</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="TACO-X vs vLLM Side-by-Side Comparison")
    parser.add_argument("--taco-url", default="http://localhost:18080/v1",
                        help="TACO-X server URL (default: http://localhost:18080/v1)")
    parser.add_argument("--taco-model", default="Qwen/Qwen3-32B",
                        help="TACO-X model name")
    parser.add_argument("--vllm-url", default="http://43.153.138.241:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--vllm-model", default="Qwen/Qwen3-32B",
                        help="vLLM model name")
    parser.add_argument("--port", type=int, default=8080,
                        help="Web UI port (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Web UI host (default: 0.0.0.0)")
    parser.add_argument("--warmup-file", default=None, metavar="FILE",
                        help="JSON file with warmup prompts for TACO-X lookahead cache")
    # Keep --base-url and --model for backwards compat
    parser.add_argument("--base-url", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    global taco_client, taco_model, vllm_client, vllm_model, warmup_file

    # Backwards compat: --base-url maps to --taco-url
    taco_url = args.base_url or args.taco_url
    taco_model = args.model or args.taco_model

    taco_client = AsyncOpenAI(base_url=taco_url, api_key="not-needed")
    vllm_client = AsyncOpenAI(base_url=args.vllm_url, api_key="not-needed")
    vllm_model = args.vllm_model
    warmup_file = args.warmup_file

    print(f"TACO-X vs vLLM Comparison")
    print(f"  Web UI:    http://localhost:{args.port}")
    print(f"  TACO-X:    {taco_url} ({taco_model})")
    print(f"  vLLM:      {args.vllm_url} ({vllm_model})")
    if warmup_file:
        print(f"  Warmup:    {warmup_file}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

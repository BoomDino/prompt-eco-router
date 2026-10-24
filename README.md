# Prompt Decomposer Eco-Router

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11–3.13-blue)
![API: FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Dashboard: Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)
![Models: Open weights](https://img.shields.io/badge/Models-Open_weights-6f42c1)
![Built for MLH Hacktoberfest](https://img.shields.io/badge/Built_for-MLH_Hacktoberfest-orange)

**An intelligent model harness and prompt-decomposing execution proxy built with open-weight AI to eliminate avoidable compute, cost, and energy waste in LLM workflows.**

Built for the **MLH Hacktoberfest Open-Source AI Prize Challenge — Best Open-Source AI Project**. This describes the project's intended entry; it does not claim organizer approval, eligibility verification, or an award.

The router accepts an OpenAI-style chat request, separates work into smaller tasks, assigns each task to a suitable model tier, validates generated output, and escalates failures. A Streamlit dashboard exposes the execution graph, latency, tokens, and estimated savings.

**Runs locally without credentials.** An empty `DO_API_KEY` enables deterministic demo responses. Add an inference key and valid model IDs to use DigitalOcean. Demo output is explicitly marked in telemetry and the dashboard; mock reasoning is not a real model answer.

## Contents

- [The problem: monolithic LLM waste](#the-problem-monolithic-llm-waste)
- [Architecture and end-to-end workflow](#architecture-and-end-to-end-workflow)
- [DigitalOcean and Hermes integration](#digitalocean-and-hermes-integration)
- [Local quickstart](#local-quickstart)
- [API reference](#api-reference)
- [Telemetry methodology](#telemetry-methodology)
- [Configuration](#configuration)
- [Verification and repository layout](#verification-and-repository-layout)
- [Limitations and deployment](#limitations-and-deployment)
- [License](#license)

## The problem: monolithic LLM waste

A prompt such as “extract errors from these logs, then propose a resilient architecture” contains two different workloads. Extracting known fields is usually much easier than reasoning about failure recovery and distributed systems. Sending both operations and all intermediate material to a large frontier model spends expensive inference capacity on inexpensive work.

Larger models can require more GPU computation, memory bandwidth, and energy per token. Repeated context processing also increases token charges. Long contexts and unnecessary generation can add latency. The exact tradeoff depends on hardware, batching, quantization, model architecture, and serving load; routing does not guarantee a faster or cheaper response.

A one-shot router makes one model selection and accepts the result. If a small model produces malformed JSON or invalid code, its apparent saving can turn into a failed workflow or a manual retry. This project adds a bounded validation-and-escalation loop, records the additional cost, and returns an explicit error if every tier fails.

Structural validation catches syntax and schema failures. It cannot establish factual accuracy, prove algorithm correctness, or detect every hallucination.

## Architecture and end-to-end workflow

```mermaid
sequenceDiagram
    actor Client as Dashboard / Hermes / HTTP client
    participant API as FastAPI ingestion
    participant Cache as In-memory fingerprint cache
    participant Planner as Rules / Hermes planner
    participant Models as DigitalOcean model tiers
    participant Verify as Output verifier
    participant Aggregate as Aggregator + telemetry
    Client->>API: POST /v1/chat/completions
    API->>Cache: Hash request + ordered prompt fingerprint
    alt Eligible duplicate cache hit
        Cache-->>API: Cached completion; zero new inference
        API-->>Client: Completion + cache telemetry
    else Cache miss
        API->>Planner: Conversation and user request
        Planner-->>API: Ordered task plan with dependencies
        loop Each task in dependency order
            API->>Models: Sub-task to Eco / Balanced / Heavy
            Models-->>Verify: Generated output + usage
            alt Structural validation fails
                Verify->>Models: Next tier + error context
                Models-->>Verify: Corrected output
            end
            Verify-->>Aggregate: Validated task result + attempts
        end
        Aggregate->>Cache: Store successful eligible completion
        Aggregate-->>Client: Chat completion + telemetry + headers
    end
```

### Step 1: Ingestion and fast semantic caching

`main.py` validates incoming requests with Pydantic. The cache computes a SHA-256 fingerprint from keywords **and the exact serialized request**. The exact guard preserves word order, punctuation, code whitespace, system instructions, prior turns, user scope, and generation settings. This is a conservative keyword-based duplicate cache, not an embedding similarity search: paraphrases do not automatically share answers.

The cache is an in-memory LRU with a default capacity of 256 responses and a 900-second TTL. Hits avoid all planner and model calls and return zero newly processed inference tokens. They still use local CPU, memory, and networking: “zero cost” means zero new model inference charges, not zero physical energy.

Only requests with `temperature: 0`, `cache: true`, and no tool conversation are cached. Streaming and non-streaming clients can reuse the same answer. Restarting the process clears the cache; workers have independent caches. Concurrent identical misses can both perform inference.

### Step 2: Prompt decomposition and sub-task planning

Two planner modes are available:

- **`PLANNER_MODE=rules` (default):** a local parser splits explicit `then`, numbered task lines, or a line containing `---`. It classifies architecture, deep reasoning, algorithms, and refactoring as Heavy; code, scripts, and tests as Balanced; otherwise Eco. Tasks run sequentially, with each subsequent task depending on the previous result. At most eight tasks are produced.
- **`PLANNER_MODE=model`:** a configured open-weight planner such as **Hermes-3-8B** or Qwen-2.5-7B produces a JSON task array. The router checks task count, fields, tier values, unique IDs, and backward-only dependencies. An invalid plan falls back to rules and records a warning. Planner tokens and costs remain in telemetry. A network or authentication failure returns an upstream error rather than pretending planning succeeded. Demo mode always uses rules.

Example plan:

```json
[
  {
    "task_id": 1,
    "prompt_segment": "Extract ERROR entries from the supplied logs as JSON.",
    "target_tier": "eco",
    "dependencies": []
  },
  {
    "task_id": 2,
    "prompt_segment": "Analyze the architecture using the extracted errors.",
    "target_tier": "heavy",
    "dependencies": [1]
  }
]
```

Each task receives its instructions, preceding conversation context, and declared dependency outputs. For a single task, the original conversation is forwarded. A model-generated plan is structurally checked but cannot be guaranteed to preserve every semantic constraint.

### Step 3: Energy-aware multi-tier dispatch

| Tier | Example open-weight models | Typical work | Illustrative Wh / 1,000 tokens |
| --- | --- | --- | ---: |
| **Eco / Nano** | Llama-3.2-3B, Gemma-2-2B | Extraction, formatting, summaries, boilerplate | 0.05 |
| **Balanced** | Qwen-2.5-Coder-7B, Hermes-3-8B | Scripts, ordinary code, tests, standard logic | 0.15 |
| **Heavy / Frontier** | Llama-3.3-70B | Architecture, algorithm refactoring, deep reasoning | 0.50 |

The example coefficients assign Eco **approximately 90% lower energy per token** than Heavy (`1 − 0.05 / 0.50`). This is an explicit modeling assumption, **not a measured benchmark or a claim about DigitalOcean hardware**. It is not an end-to-end saving guarantee. Configure coefficients and model IDs for the deployment you actually use.

The harness uses HTTPX to call a shared OpenAI-compatible inference endpoint with the model ID selected for each tier. These example families are open-weight, but each model retains its own license. The repository's MIT license does not relicense model weights.

### Step 4: Speculative verification and auto-escalation

The router first tries the assigned tier, then uses a larger tier only when validation fails:

```text
Eco output invalid → Balanced + validation error → Heavy + validation error
Balanced output invalid → Heavy + validation error
Heavy output invalid → HTTP 502; do not cache the failed result
```

Validation checks:

- Nonempty text for ordinary responses.
- JSON parsing when the task asks for JSON.
- A JSON object for `response_format: {"type": "json_object"}`.
- JSON Schema conformance for `response_format.type: "json_schema"`.
- Python syntax with `ast.parse` when Python is requested; generated code is never executed.
- A provider `finish_reason: "length"` triggers a shorter-answer retry at the next tier.

Each retry receives the validation error as correction context. This is speculative execution in the sense of trying a cheaper tier first; it does not run multiple model tiers concurrently. Fallback is automatic for the caller and visible in telemetry. HTTP errors, invalid credentials, and rate limits are surfaced instead of silently substituting mock answers.

Explicit structured-output requests remain one task so the final result can satisfy the caller's schema as a whole. JSON syntax validation alone does not prove the extracted data is correct.

### Step 5: Aggregation and live eco telemetry

Single-task responses are returned directly. Multiple outputs are merged deterministically into ordered `Task N` sections. Dependency-aware reasoning happens within the selected task models; aggregation adds no hidden synthesis-model call.

The response preserves standard chat-completion fields and adds `eco_telemetry`. Custom headers expose savings to clients that do not inspect extension fields. The dashboard updates after each completed request, displays actual final tiers and escalations, and offers a raw request/response toggle. It does not display live token-by-token execution events.

## DigitalOcean and Hermes integration

### DigitalOcean AI inference

The inference path uses:

```text
POST https://inference.do-ai.run/v1/chat/completions
Authorization: Bearer <DO_API_KEY>
```

DigitalOcean documents separate inference and control-plane APIs. See the [inference endpoint reference](https://docs.digitalocean.com/products/inference/how-to/si-endpoints/) and [supported model catalog](https://docs.digitalocean.com/products/inference/details/models/).

Set `DO_INFERENCE_ENDPOINT` to the base URL **including `/v1`**, without `/chat/completions`. The app adds the final route. Set `DO_API_KEY` to a credential accepted by the inference service.

**Confirm your model IDs before enabling real inference.** The example 2B–8B families and Hermes planner are not guaranteed to be available in the serverless catalog. Query the endpoint:

```bash
curl https://inference.do-ai.run/v1/models \
  -H "Authorization: Bearer $DO_API_KEY"
```

Then configure `DO_MODEL_ECO`, `DO_MODEL_BALANCED`, `DO_MODEL_HEAVY`, and optionally `DO_PLANNER_MODEL`. Choose available open-weight models or provision a compatible dedicated deployment. The current router uses one shared endpoint and credential for all tiers. A dedicated deployment or gateway must expose every configured model there. It does not provision model servers or download weights automatically.

### DigitalOcean libraries, resource management, and monitoring

- **HTTPX:** the inference transport, with connection pooling, timeouts, and bounded validation retries.
- **PyDo:** DigitalOcean's official Python SDK, integrated through the optional, read-only `scripts/cloud_resources.py` utility. It inventories Droplets and their region/size/status to help operators track resources hosting a router or agent.
- **Application telemetry:** token usage and estimated energy/cost are computed inside the router and returned with each request. They are not retrieved from cloud billing or a physical GPU power meter.

To use the SDK helper:

```bash
pip install -r requirements-cloud.txt
export DIGITALOCEAN_TOKEN='your-control-plane-token'
python scripts/cloud_resources.py
```

The helper lists resources; it does not create or delete them. `DIGITALOCEAN_TOKEN` is separate from `DO_API_KEY`. For deployment infrastructure metrics, use DigitalOcean Monitoring on the hosting Droplet; this project does not install that agent or ingest its metrics. See [PyDo documentation](https://docs.digitalocean.com/reference/pydo/) for cloud resource APIs. There is no claim that PyDo supplies serverless model energy measurements.

### One-Click Hermes Agent compatibility

The sponsor-provided [DigitalOcean Hermes One-Click link](https://mlh.link/digitalocean-hermes-h26) is the starting point for the supplied deployment offering. Its destination and provisioning flow were not independently verified during this build; follow the current page and organizer instructions.

**Hermes Agent** is the agent runtime. **Hermes-3-8B** is an optional planner/model family. They are distinct components.

Once Hermes Agent is running, point its custom OpenAI-compatible provider at this router. The official [Hermes provider guide](https://hermes-agent.nousresearch.com/docs/integrations/providers/) documents custom endpoint setup through `hermes model`.

1. Start the router on a network reachable from the agent.
2. Run `hermes model` and select **Custom endpoint**.
3. Set the base URL to `http://127.0.0.1:8000/v1` when both processes share the same host.
4. Select model **`eco-router`**.
5. Set the client API key to the router's `ROUTER_API_KEY`, or a dummy local value if router authentication is disabled. Keep the DigitalOcean inference key on the router server.

Equivalent model configuration for Hermes versions supporting these fields:

```yaml
model:
  provider: custom
  default: eco-router
  base_url: http://127.0.0.1:8000/v1
  api_key: local-demo
```

For a remote One-Click Droplet, `127.0.0.1` refers to that Droplet, not your laptop. Deploy the router alongside it or use a reachable protected URL/tunnel, and configure the appropriate key. Consult the installed Hermes version's provider guide if its config format differs.

**Agent protocol behavior:**

- Text-only turns can use normal decomposition and caching.
- Requests with tool definitions, assistant tool calls, or tool-result messages are forwarded as an intact conversation to Heavy. This preserves tool IDs, arguments, and result ordering.
- Tool conversations bypass the cache and decomposition, preventing replay of a previously selected external action.
- Hermes executes tools; the router never executes them.
- `stream: true` returns SSE chat-completion chunks, including indexed tool calls and `[DONE]`. Generation is buffered until routing and validation finish. This is wire compatibility, not reduced time to first token.
- Real tool-call support depends on the selected upstream model. Demo mode supplies text only and does not simulate an autonomous agent loop.

This implements the HTTP adapter surface. An actual One-Click Hermes deployment and paid inference run still require credentials and integration verification in your environment.

## Local quickstart

### 1. Clone and create a Python environment

Use Python **3.11–3.13**, preferably **3.12**, with these dependency pins. Python 3.14 is not a supported target for this pinned environment. Replace the example repository URL with the URL where you publish this project:

```bash
git clone https://github.com/YOUR_USERNAME/prompt-eco-router.git
cd prompt-eco-router
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

### 2. Install pinned dependencies

```bash
pip install -r requirements.txt
cp .env.example .env
```

PowerShell users can use `Copy-Item .env.example .env`. Direct runtime dependencies are pinned; transitive dependencies are resolved by pip. The optional cloud SDK has a separate requirements file.

Leave `DO_API_KEY` empty for the self-contained demo. `.env` loads automatically and is excluded from Git.

### 3. Start FastAPI

```bash
uvicorn main:app --reload
```

- Health: <http://127.0.0.1:8000/health>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Model discovery: <http://127.0.0.1:8000/v1/models>

### 4. Start Streamlit in another terminal

Activate the same environment and run:

```bash
streamlit run ui.py
```

Open <http://localhost:8501>. Select one of the three presets, click **Decompose & route**, and inspect metrics, task details, and raw JSON. The sidebar configures the router URL and optional router key. API errors appear in the dashboard with a useful status message.

### 5. Enable actual model inference

Edit `.env`, supply `DO_API_KEY`, and set model IDs confirmed by your endpoint. Restart the backend after changing environment values; reload primarily watches source files. Set `PLANNER_MODE=model` only after configuring an accessible planner model.

## API reference

### Text completion

```bash
curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "eco-router",
    "messages": [{"role": "user", "content": "Format these names as bullets: Ada, Grace, Linus"}],
    "temperature": 0,
    "max_tokens": 1024
  }'
```

If authentication is enabled, add `-H "Authorization: Bearer $ROUTER_API_KEY"`.

Response structure, with values abbreviated for readability:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1792800000,
  "model": "eco-router",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "- Ada\n- Grace\n- Linus"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37},
  "eco_telemetry": {
    "mock_mode": true,
    "cache_hit": false,
    "processed_tokens": 37,
    "measurement": "Illustrative estimates; not measured power or provider billing."
  }
}
```

The actual `eco_telemetry` also includes latency, baseline values, savings, model rates, planner status, tasks, dependencies, outputs, and per-attempt usage/validation errors.

### Structured output

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "eco-router",
    "messages": [{"role": "user", "content": "Extract ERROR lines: ERROR database timeout"}],
    "response_format": {"type": "json_object"}
  }'
```

For JSON Schema, use `{"type":"json_schema","json_schema":{"name":"result","schema":{"type":"object"}}}` as `response_format`. Local validation uses JSON Schema Draft 2020-12. The upstream service must support the requested format. Demo mode returns a fixed `{mock, tier, items}` object; an incompatible custom schema intentionally fails validation rather than fabricating compliance.

### Supported request fields

| Field | Behavior |
| --- | --- |
| `model` | Virtual model `eco-router`; upstream IDs are server configuration |
| `messages` | Text content, system/developer/user/assistant/tool roles, tool IDs and calls |
| `temperature` | 0–2; default 0; cache disabled above 0 |
| `max_tokens`, `max_completion_tokens` | 1–8192 per model call; latter takes precedence; not a whole-plan budget |
| `response_format` | `text`, `json_object`, or `json_schema` |
| `tools`, `tool_choice`, `parallel_tool_calls` | Forwarded on agent requests |
| `stop`, `seed` | Forwarded to upstream |
| `stream`, `stream_options.include_usage` | Buffered SSE response; optional usage chunk |
| `n` | Only 1 supported |
| `user` | Included in cache isolation; not forwarded upstream |
| `cache` | Router extension; default true |

Unknown fields are rejected, not silently ignored. Multimodal content arrays, legacy `functions`, logprobs, and the Responses API are not implemented. This is a defined Chat Completions subset, not complete OpenAI API parity. Conversations are limited to 100 messages and 100,000 text characters; real provider context limits may be smaller.

### Errors

Errors use an `error` object with a message and type:

| HTTP status | Meaning |
| --- | --- |
| 401 | Missing/incorrect router key when authentication is enabled |
| 422 | Unsupported or invalid request/schema |
| 429 | Upstream rate limit; no automatic network retry |
| 502 | Upstream failure, malformed response, or final validation failure |
| 504 | Upstream timeout |

No fallback to mocks occurs after a real inference request fails. Failed completions are not cached. An upstream authentication failure becomes a sanitized 502 so it is distinguishable from router authentication.

## Telemetry methodology

The system estimates a counterfactual: **one Heavy call with the original conversation and approximately the same final output**, compared with all calls actually made by the router. It does not call Heavy solely to measure a baseline.

For each attempt `i` at tier `t`:

```text
T_i = prompt_tokens_i + completion_tokens_i
estimated_energy_Wh = sum(T_i × Wh_per_1000_tokens[t] / 1000)
estimated_cost_USD  = sum(T_i × USD_per_million_tokens[t] / 1,000,000)

baseline_tokens = estimate(original_messages) + estimate(final_assistant_message)
baseline_Wh     = baseline_tokens × Heavy_Wh_per_1000 / 1000
baseline_USD    = baseline_tokens × Heavy_USD_per_million / 1,000,000

Wh_saved  = baseline_Wh  − estimated_energy_Wh
USD_saved = baseline_USD − estimated_cost_USD
```

Provider token counts are used when available; otherwise `max(1, ceil(UTF-8 bytes / 4))` is used. The baseline always uses that approximation. Planner calls, correction prompts, dependency text, and unsuccessful generated attempts are included. Input/output tokens share one illustrative rate per tier. No provider-specific tokenizer, separate input/output pricing, cached-input tariff, hardware measurement, or carbon-intensity model is included.

Savings are signed: negative values mean the modeled route costs more. This avoids overstating benefits for tiny prompts, repeated retries, or expensive planning. In mock mode all values are simulated; actual cloud spend is zero. On a cache hit, new usage, modeled inference energy, and modeled inference cost are zero; the saved amount equals the cached baseline. Cached task traces describe the original plan and are marked `cached: true`; current attempts are empty.

Headers:

- `X-Eco-Wh-Saved`: signed estimated Watt-hours saved.
- `X-Eco-Cost-Saved`: signed estimated US dollars saved.
- `X-Eco-Cache-Hit`: `true` or `false`.

These are successful-response estimates. A failed HTTP request may still incur provider charges; failure responses do not currently return partial usage accounting.

## Configuration

See [.env.example](.env.example) for a complete editable template.

| Variable | Default / purpose |
| --- | --- |
| `DO_API_KEY` | Empty enables demo mode |
| `DO_INFERENCE_ENDPOINT` | `https://inference.do-ai.run/v1` |
| `DO_MODEL_ECO` | `llama-3.2-3b-instruct` example ID |
| `DO_MODEL_BALANCED` | `qwen2.5-coder-7b-instruct` example ID |
| `DO_MODEL_HEAVY` | `llama3.3-70b-instruct` example ID |
| `DO_PLANNER_MODEL` | `hermes-3-llama-3.1-8b` example ID |
| `PLANNER_MODE` | `rules` or `model` |
| `ROUTER_API_KEY` | Optional Bearer authentication for `/v1/*` |
| `ECO_ROUTER_URL` | Dashboard target, default `http://127.0.0.1:8000` |
| `CACHE_MAX_ENTRIES` | 256; 0 disables storage |
| `CACHE_TTL_SECONDS` | 900 |
| `UPSTREAM_TIMEOUT_SECONDS` | 90 per upstream call |
| `DEMO_FAIL_ECO` | `false`; `true` makes mock Eco output empty to demonstrate escalation |
| `{ECO,BALANCED,HEAVY}_WH_PER_1K` | Illustrative energy coefficients: 0.05 / 0.15 / 0.50 |
| `{ECO,BALANCED,HEAVY}_USD_PER_MILLION` | Illustrative dollar coefficients: 0.10 / 0.30 / 1.00 |
| `DIGITALOCEAN_TOKEN` | Optional PyDo control-plane token |

## Verification and repository layout

Run the local integration checks:

```bash
python -m unittest discover -s tests -v
```

Checks use FastAPI's test client and HTTPX mock transports. They cover routing, cache isolation, validation escalation, failure handling, structured output, agent tool calls, SSE, and authentication without external API charges.

```text
.
├── main.py                    # FastAPI routes, auth, errors, SSE
├── ui.py                      # Streamlit presets, graph, metrics, raw JSON
├── eco_router/
│   ├── config.py              # Environment configuration and assumptions
│   ├── models.py              # Validated request and task schemas
│   ├── cache.py               # Bounded request fingerprint cache
│   ├── planner.py             # Rule parser and model-plan validation
│   ├── provider.py            # HTTPX DigitalOcean client and demo backend
│   ├── validation.py          # JSON/schema/Python syntax checks
│   └── engine.py              # Orchestration, escalation, aggregation, accounting
├── scripts/cloud_resources.py # Optional PyDo resource inventory
├── tests/test_router.py       # Network-free integration checks
├── requirements.txt           # Pinned direct runtime dependencies
├── requirements-cloud.txt     # Optional pinned DigitalOcean SDK
├── .env.example
├── .gitignore
└── LICENSE
```

## Limitations and deployment

This is a complete local hackathon harness with explicit production boundaries:

- Heuristic routing and syntax checks do not guarantee model quality. Treat outputs as untrusted content.
- Tasks execute sequentially. The router does not schedule independent DAG branches concurrently.
- Dependency outputs and earlier conversation turns can still be large. There is no context-window optimizer or tokenizer-aware truncation.
- Aggregation uses ordered sections; it does not perform an additional global reasoning pass.
- The cache is process-local and can retain prompt-derived answers in memory until eviction. It has no shared storage or per-tenant access management beyond optional key/user scoping.
- SSE is buffered. The dashboard's metrics refresh at request completion.
- Model names and energy coefficients are examples; real deployment requires catalog checks and calibrated measurements.
- Arbitrary generated code is never executed by the verifier. Python parsing proves syntax only.

For deployment, run Uvicorn without `--reload`, set `ROUTER_API_KEY`, and place public services behind TLS and appropriate request-size/rate limits. Use managed secrets or server environment variables. Multiple workers improve capacity but maintain separate caches. The health endpoint verifies the process, not inference credentials or model availability. The optional PyDo helper does not deploy infrastructure.

For troubleshooting: a connection error usually means the backend URL or port is wrong; upstream 401/403 means check the inference credential; upstream 404 often means an incorrect model or base URL; a schema failure in demo mode may simply require a schema matching the documented mock object. Restart after changing `.env`.

## License

This project is open-source under the **MIT License**. See [LICENSE](LICENSE) for the complete terms. Model weights, DigitalOcean services, and Hermes Agent retain their respective licenses and service terms.

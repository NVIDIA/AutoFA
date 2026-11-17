# Openclaw — Architecture & loop

**At a glance:** A single Gateway daemon owns all channels and exposes a typed WebSocket API (`127.0.0.1:18789`). An `agent` RPC starts a run that is serialized through per-session + global queues, executed by `runEmbeddedAgent`, and streamed back as `tool` / `assistant` / `lifecycle` events. Loop termination is **stop-condition + timeout driven — no documented hard max-iteration cap.**

## How it works

**Process model** (`/concepts/architecture`):
- One **Gateway daemon** owns every messaging surface (WhatsApp via Baileys, Telegram via grammY, …) and exposes a **typed WebSocket API**, default bind `127.0.0.1:18789`.
- Clients (macOS app, CLI, web UI) and Nodes (`role: node`) connect over that WebSocket.

**Agent turn control flow** (`/concepts/agent-loop`):
1. `agent` RPC validates params, resolves the session, returns `{ runId, acceptedAt }` immediately (async).
2. `agentCommand` loads the **skills snapshot** and calls `runEmbeddedAgent`.
3. `runEmbeddedAgent` serializes runs via **per-session and global queues**, resolves model + auth profile, builds the Openclaw session, subscribes to runtime events, streams assistant/tool deltas, and **enforces the run timeout**.
4. `subscribeEmbeddedAgentSession` bridges runtime events to three streams: **`tool`**, **`assistant`**, **`lifecycle`**.

**Harness layer** (`/plugins/sdk-agent-harness`): the harness executes *one prepared turn* after the Gateway has resolved provider/model/auth/thinking/context/transcript/workspace/sandbox/tool-policy/channel callbacks.

**Stop conditions:** agent run timeout (abort), `AbortSignal` (cancel), Gateway disconnect / RPC timeout, or lifecycle end/error from the embedded loop. `agent.wait` is wait-only and does **not** stop the underlying run. **Events are not replayed** — clients refresh on gaps; a short-lived dedupe cache gives idempotency.

## Where errors can originate

- **No documented hard max-iteration / max-tool-call cap** — loop termination is model/stop-driven plus timeouts, so a model that never emits a final answer runs until the **run timeout** fires (looks like a hang/timeout in the trace, not a clean stop). *(documented gap: no explicit turn-count limit.)*
- **Run serialization queues** (per-session + global): a long-running or stuck run blocks queued runs for the same session — throughput stalls that present as latency, not errors.
- **Event gaps desync observers:** because events are not replayed, a dropped WebSocket event forces a client/observer refresh; a trace reconstructed from events can miss intermediate state.
- **Fail-hard runtime selection** (see [error-taxonomy.md](error-taxonomy.md)): once a plugin harness claims a run, the turn is **not replayed** through another runtime.

## Sources

- `https://docs.openclaw.ai/concepts/agent-loop` — `agent` RPC, `runEmbeddedAgent`, queues, run timeout, stop conditions.
- `https://docs.openclaw.ai/concepts/architecture`, `https://docs.openclaw.ai/concepts/agent-runtimes` — Gateway process model, runtime/harness distinction.

> Gap: no hard per-turn max-iteration / max-tool-call limit is documented — only timeouts + stop conditions.

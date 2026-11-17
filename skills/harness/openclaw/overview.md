# Openclaw — Overview

**At a glance:** Openclaw is a self-hosted, local-first **Gateway** (TypeScript/Node) that bridges chat channels (WhatsApp, Telegram, Slack, Discord, Signal, iMessage, IRC, Teams, Matrix, WebChat…) to locally-run AI agents. The "agent harness" is one *internal layer* of it — the per-turn executor — not a standalone coding-benchmark runner.

## How it works

- **What it is.** A personal-assistant gateway: one long-lived Gateway process is the central hub that connects messaging surfaces to AI agents. Supports multi-agent routing, voice, media, a browser "Canvas," and mobile "nodes." Marketed as "Your own personal AI assistant. Any OS. Any Platform. The lobster way. 🦞"
- **Purpose.** Local-first / privacy-preserving assistant reachable from chat apps you already use.
- **Runtime / language.** **TypeScript on Node.js.** Requires Node 24.15+ / 22 LTS (22.22.3+) / 25.9+. Distributed as a global **npm** package; build tooling uses **pnpm**; protocol types are **TypeBox** schemas (JSON Schema + generated Swift models); "code mode" embeds a **QuickJS-WASI** JS VM.
- **Layering that matters for eval** (`/concepts/agent-runtimes`): **Provider → Model → Agent runtime → Harness**.
  - *Agent runtime* = the loop/backend that executes a prepared turn; it "receives the prompt, drives model output, handles native tool calls, and returns the finished turn," and **owns where retries, tool continuation, and final-answer decisions happen.**
  - *Harness* = the code implementing a runtime — "the low-level executor for one prepared Openclaw agent turn," invoked *after* Openclaw has already resolved provider, model, auth, thinking level, context budget, transcript/session file, workspace, sandbox, tool policy, and channel callbacks.
- Because so much is resolved *before* the harness runs, many "harness" outcomes in a trace actually trace back to the Gateway's pre-turn resolution (tool policy, context budget, compaction) — see [error-taxonomy.md](error-taxonomy.md).

## Where errors can originate

- **Mis-attribution risk:** what looks like a harness bug in a trace may be a Gateway-layer decision (tool policy stripping, compaction, model failover) made *before* the per-turn executor ran. Always locate the failure in the layer stack above before blaming the executor.
- **Runtime version drift:** strict Node version floors (24.15+/22.22.3+/25.9+) mean a mismatched host can change behavior or fail to start.

## Sources

- `https://docs.openclaw.ai/concepts/agent-runtimes` — Provider→Model→Runtime→Harness layering, what the runtime owns.
- `https://docs.openclaw.ai/concepts/architecture`, `https://docs.openclaw.ai/plugins/sdk-agent-harness` — Gateway process model, harness = per-turn executor.

> Note: public "star count / most-starred repo" claims are unverified marketing and are intentionally omitted.

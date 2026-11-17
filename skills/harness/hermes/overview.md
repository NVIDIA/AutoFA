# Hermes — Overview

**At a glance:** Hermes is a self-improving personal AI agent (Python) with one shared "agent core" driven across a CLI, a ~20-platform messaging gateway, a TUI, and an Electron desktop app. The core is a deliberately **narrow waist**: every model tool ships on every API call, and per-conversation prompt caching is treated as an invariant.

## How it works

- **What it is.** A model-agnostic agent core (`AIAgent` in `run_agent.py`) with: a learning loop (auto skill creation + self-improving skills), cross-session memory (SQLite FTS5 session search + optional Honcho/Hindsight/Mem0 providers), cron automations, subagent delegation, six terminal backends (local/Docker/SSH/Singularity/Modal/Daytona), and MCP support.
- **Purpose.** "The self-improving AI agent built by Nous Research" — a personal agent that grows with the user across surfaces.
- **Runtime / language.** **Python** (`requires-python = ">=3.11,<3.14"`), packaged as `hermes-agent`. Console entry points: `hermes = "hermes_cli.main:main"` and `hermes-acp = "acp_adapter.entry:main"`. CLI stack: `fire==0.7.1`, `prompt_toolkit`, `rich`. LLM transport: `openai==2.24.0` SDK + `httpx[socks]`; retries via `tenacity`; templating via `jinja2`. TypeScript/Electron lives in `tui_gateway/`, `ui-tui/`, `web/`, `website/`, `apps/`. MIT license.
- **Design invariants (from `AGENTS.md`)** — these directly shape failure modes:
  - **"Per-conversation prompt caching is sacred."** The system prompt is built once per session and reused every turn; only context compression may rebuild it (see [prompt-and-context.md](prompt-and-context.md)).
  - **"The core is a narrow waist — every model tool ships on every API call."** Tools are not dynamically stripped per turn (contrast with Openclaw's pre-call policy stripping).
  - **`.env` = secrets only; `~/.hermes/config.yaml` = all behavior** (see [configuration.md](configuration.md)).
- **Dependency hardening.** Every direct dependency is exact-pinned (`==X.Y.Z`) specifically to prevent a transitive being swapped underneath — a supply-chain rationale documented in `pyproject.toml`.

## Where errors can originate

- **Version floor mismatch:** Python must be `>=3.11,<3.14`; running outside that range is unsupported.
- **Mis-attribution risk:** because the same `AIAgent` core serves every surface (CLI/gateway/TUI/desktop), a surface-specific system-prompt template (WhatsApp vs CLI, see [prompt-and-context.md](prompt-and-context.md)) can change behavior for the "same" task — check which surface produced the trace.

## Sources

- `nousresearch/hermes-agent` `AGENTS.md` — design invariants (prompt-cache sacred, narrow-waist tools, `.env` vs config split).
- `nousresearch/hermes-agent` `README.md`, `pyproject.toml` — what it is, entry points, dependency pinning.

> Note: earlier auto-fetched repo "stats" (star / commit / version counts) were summarizer inventions and are intentionally excluded — do not treat any such number as fact; `/llms.txt` 404s and is not used.

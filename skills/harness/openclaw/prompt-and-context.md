# Openclaw — Prompt & context

**At a glance:** `buildAgentSystemPrompt` renders a **fixed-order** system prompt tuned for **prefix prompt-caching** (stable content above the cache boundary, volatile below). Workspace bootstrap files are injected under hard char caps (20k/file, 60k total). Context is kept in budget by **auto-compaction** that keeps the most recent ~20k tokens and can trigger a retry.

## How it works

**System-prompt builder** (`/concepts/system-prompt`):
- `buildAgentSystemPrompt` is a **pure renderer**; `resolveAgentSystemPromptConfig` resolves config knobs (owner display, TTS hints, model aliases, memory citation mode, sub-agent delegation mode).
- **Fixed section order:** Tooling → Execution Bias → Safety → Skills → Openclaw Control → Self-Update → Workspace → Documentation → Workspace Files → Sandbox → Current Date/Time → Assistant Output Directives → Heartbeats → Runtime → Reasoning.
- **Prompt-cache strategy:** large stable content sits **above** the prompt-cache boundary; volatile sections (messaging, voice, reactions, runtime) are appended **below** so prefix caches reuse the stable prefix. The Date/Time section carries **timezone only, no live clock** (cache stability); live time comes from `session_status`.
- **Provider overrides:** a provider runtime may replace exactly one of three named core sections — `interaction_style`, `tool_call_style`, `execution_bias` — plus inject a stable prefix / dynamic suffix.
- **Bootstrap file injection:** `AGENTS.md`, `SOUL.md`, `TOOLS.md`, `IDENTITY.md`, `USER.md`, `HEARTBEAT.md`, `BOOTSTRAP.md` (new workspaces only), `MEMORY.md`. Per-file cap `agents.defaults.bootstrapMaxChars` (default **20,000**); total `agents.defaults.bootstrapTotalMaxChars` (default **60,000**). Sub-agents get only `AGENTS.md` + `TOOLS.md`.
- **Prompt modes** (`promptMode`): `full` (default), `minimal` (sub-agents; drops memory/self-update/identity/etc.), `none` (base identity line only).
- **Memory:** `memory/*.md` daily files are on-demand via `memory_search` / `memory_get` — **not counted against context unless read**; excerpt budgets under `agents.defaults.contextLimits.*`.

**Compaction / context budget** (`/concepts/compaction`):
- Auto-compaction triggers when the session nears the limit **or** the model returns a context-overflow error → Openclaw **compacts and retries**.
- Summarizes older turns, keeps recent (`agents.defaults.compaction.keepRecentTokens`, default **20,000**); mode default `"safeguard"`; also `maxActiveTranscriptBytes` and a delegate `compaction.model`.

## Where errors can originate

- **Bootstrap truncation:** files are truncated at 20k/60k chars with a warning (`bootstrapPromptTruncationWarning` default `always`) — **silent context loss** if `AGENTS.md`/`USER.md`/etc. are large. Model appears to "forget" instructions that were cut.
- **Compaction drops detail:** summarization of older turns can lose task-critical facts; manual `/compact` without `keepRecentTokens` acts as a **hard checkpoint** that drops detail.
- **Retry-induced duplication/suppression:** compaction can trigger a retry; on retry, in-memory buffers/tool summaries reset "to avoid duplicate output," and successor transcripts **drop exact-duplicate long user turns** inside a short retry window — dedupe that can both suppress and mask real content.
- **Date staleness:** the prompt's Date/Time carries timezone only; a model that needs the wall clock must call `session_status` — otherwise it may reason with a stale/absent time.
- **Prompt-mode downgrade:** sub-agents run in `minimal` mode (no memory/identity) — a sub-agent may behave differently from the top-level agent on the same task.

## Sources

- `https://docs.openclaw.ai/concepts/system-prompt` — builder, section order, cache strategy, bootstrap files + caps, prompt modes, memory.
- `https://docs.openclaw.ai/concepts/compaction` — auto-compaction triggers, `keepRecentTokens=20000`, `safeguard` mode, retry behavior, `/compact` checkpoint.

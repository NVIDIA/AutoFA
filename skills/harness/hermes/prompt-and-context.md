# Hermes — Prompt & context

**At a glance:** The system prompt is built **once per session** and reused every turn (cache-warmth invariant); only compression rebuilds it. It joins three tiers — **stable** (identity/tools/skills), **context** (caller message + discovered `AGENTS.md`/`HERMES.md`/etc.), **volatile** (memory/user profile/model line). Context is managed by a pluggable compression engine that compresses at **75%** of the window; Anthropic prompt caching uses a `system_and_3` breakpoint layout.

## How it works

**Assembly** joins three tiers with `\n\n`:
- **stable** — identity (`SOUL.md` or default identity string), tool guidance, Nous-subscription block, per-model operational guidance, **skills index**, environment/platform hints.
- **context** — caller `system_message` + discovered context files (`AGENTS.md`/`HERMES.md`/`CLAUDE.md`/`.cursorrules`) under `TERMINAL_CWD`.
- **volatile** — memory snapshot, `USER.md` profile, external-memory provider block, and a trailing `Model: …\nProvider: …` identity line.

**Per-surface templates.** Platform-specific system text (WhatsApp/Telegram/Discord/Slack/Signal/email/cron/CLI/TUI/desktop/SMS/iMessage/Matrix/…) is templated in `prompt_builder.py`. The skills index is snapshotted/cached. `/learn` builds a one-shot skill-authoring prompt.

**Context management (pluggable).** A `ContextEngine` abstraction handles `on_session_start → update_from_response → should_compress → compress → on_session_end`; default engine `"compressor"` (config `context.engine`). Memory is pluggable (one external provider at a time).

**Prompt caching.** Anthropic `system_and_3` layout — 4 `cache_control` breakpoints (system + last 3 non-system messages), TTL `5m` or `1h`; handles native-Anthropic vs OpenRouter envelope quirks.

**Key defaults.** Compress at **75%** of context, `protect_first_n = 3`, `protect_last_n = 6`. Compression requires **>5% token reduction** to count as progress. Skill descriptions are hard-limited to **60 chars** in the index.

## Where errors can originate

- **Cache invalidation / cost blowup:** any mid-conversation mutation of past context, a toolset swap, or a system-prompt rebuild breaks the cached prefix — `AGENTS.md` calls this the cardinal sin; compression is the only sanctioned exception.
- **Context overflow → forced compression/reset:** if compression can't make >5% progress it can surface "Cannot compress further" / auto-reset.
- **Preflight undercount:** the count-only preflight gate omits the system prompt + tool schemas, so a few very large messages can slip past it — it's a hint, not exact.
- **Skill-index silent truncation:** descriptions past 60 chars are cut and may never route.
- **Context-file truncation:** files are capped by window size; oversized files are truncated with a warning.

## Sources

- `nousresearch/hermes-agent` `agent/system_prompt.py`, `agent/prompt_builder.py` — three-tier assembly, per-surface templates, context-file truncation, skills index.
- `nousresearch/hermes-agent` `agent/context_engine.py`, `agent/prompt_caching.py` — compressor engine, 0.75 threshold, Anthropic cache layout.

> Partial read: `context_compressor.py` and `conversation_compression.py` concrete internals were not deep-read.

# Hermes — Configuration

**At a glance:** Two-layer, deliberately separated config: **`.env` holds secrets only** (API keys/tokens); **`~/.hermes/config.yaml` holds all behavior** (model, timeouts, thresholds, feature flags). Adding a `HERMES_*` env var for non-secret behavior is an explicit anti-pattern. Home dir `~/.hermes` (`%LOCALAPPDATA%\hermes` on Windows).

## How it works

**Layering** (from `AGENTS.md`): `.env` = secrets only; `~/.hermes/config.yaml` = all behavior; `.env` takes precedence for secrets. Config surfaces: `hermes config set/get`, `hermes model`, `hermes tools`, `hermes setup [--portal]`.

**Model block** (`cli-config.yaml.example`):
- `model.default` (example `"anthropic/claude-opus-4.6"`), `model.provider` (`"auto"` or one of nous/openrouter/anthropic/openai-codex/copilot/gemini/zai/kimi-coding/minimax/nvidia/xiaomi/arcee/ollama-cloud/deepinfra/azure-foundry/lmstudio/custom…), `model.base_url` (default `https://openrouter.ai/api/v1`), `model.context_length` (auto-detected), `model.max_tokens` (unset → native ceiling), `model.default_headers`/`extra_headers`.

**Per-provider overrides** (`cli-config.yaml.example`): `providers.<id>.request_timeout_seconds`, `.stale_timeout_seconds`, `.models.<model>.timeout_seconds`, `.extra_headers`.

**Other behavior sections:** `provider_routing.sort` (price/throughput/latency), `openrouter.response_cache` / `response_cache_ttl` (300 s), `worktree`/`worktree_sync`, `context.engine` (`"compressor"`), `memory.provider`, `moa.save_traces`/`moa.trace_dir`, `tool_loop_guardrails.{warn_after,hard_stop_after,…}`, `delegation.max_iterations` (50), `max_iterations` (90).

**Behavioral env vars (mostly legacy/bridged):** `HERMES_HOME`, `HERMES_API_TIMEOUT`=1800, `HERMES_API_CALL_STALE_TIMEOUT`=90, `HERMES_STREAM_STALE_TIMEOUT`=180, `HERMES_STREAM_STALE_GIVEUP`=5, `HERMES_CONCURRENT_TOOL_TIMEOUT_S`=420, `HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS`, `HERMES_KANBAN_TASK`.

**Secret env vars:** API keys for all supported providers (OpenRouter, Fireworks, Novita, Google/Gemini, Ollama, GLM, Kimi, MiniMax, HuggingFace, DeepInfra, Nous, Anthropic, NVIDIA, Xiaomi, Arcee, and others) — see `.env.example` in the repo for the full list.

## Where errors can originate

- **Wrong `context_length` auto-detection** (local server / proxy hiding `/v1/models`) → mis-timed compression or request-validation failures.
- **`.env` vs `config.yaml` confusion:** a behavioral setting placed in `.env`, or a new `HERMES_*` var for non-secret config, may be ignored (explicitly discouraged in `AGENTS.md`).
- **`max_tokens` / `stale_timeout_seconds` misconfig:** too-low `max_tokens` truncates output; too-low `stale_timeout_seconds` kills reasoning models mid-think (see [model-interaction.md](model-interaction.md)).

## Sources

- `nousresearch/hermes-agent` `cli-config.yaml.example` — full behavior config reference.
- `nousresearch/hermes-agent` `AGENTS.md` — `.env` vs `config.yaml` rubric, env var anti-pattern.

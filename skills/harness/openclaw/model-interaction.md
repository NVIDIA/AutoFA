# Openclaw — Model interaction

**At a glance:** Providers are plugins that own request normalization and expose one of two API shapes — `anthropic-messages` or `openai-completions`. Transport is SSE by default (`sse`/`websocket`/`auto`). **Retries rotate to the next API key *only* on rate-limit responses; any non-rate-limit failure fails immediately with no key rotation.** Model fallback is a configured `primary` + ordered fallback chain.

## How it works

**API formats** (`/concepts/model-providers`, `/providers`):
- Provider-selected: `api: "anthropic-messages"` (Anthropic Messages API) and `api: "openai-completions"` (OpenAI Chat Completions / Responses).
- Each bundled plugin owns request normalization, model catalog, auth env mapping, tool-schema cleanup, and failover classification.
- Model refs are `provider/model` (e.g. `anthropic/claude-opus-4-8`).

**Transport / streaming:**
- `params.transport` = `"sse"` | `"websocket"` | `"auto"`; **SSE streaming is the default** (debuggable via `OPENCLAW_DEBUG_SSE`).
- Block streaming: `agents.defaults.blockStreamingDefault` (off by default), tuned by `agents.defaults.blockStreamingBreak` (`text_end` vs `message_end`).

**Token control:** `contextWindow`, `contextTokens`, `maxTokens` at provider or per-model level.

**Retries / failover** (`/concepts/model-failover`):
- Requests retry with the **next API key only on rate-limit responses** (`429`, `rate_limit`, `quota`, `resource exhausted`, `ThrottlingException`, and similar).
- **Non-rate-limit failures fail immediately — no key rotation.**
- Model **fallback chain** via `agents.defaults.model.primary` + ordered fallbacks.

**Compatibility knobs:** `compat.supportsDeveloperRole` (forced `false` on non-native endpoints); reasoning via `/think adaptive`; per-model overrides under `agents.defaults.models["<provider>/<model>"].params.*` (including `extra_body` for vendor fields).

## Where errors can originate

- **No retry on transient non-rate-limit errors:** a one-off 500 / connection reset / timeout from the provider **fails the turn immediately** (no key rotation, no retry) — shows up as an abrupt provider error mid-run.
- **Failover-chain misconfig:** an empty or wrong `model.primary` + fallbacks list means no graceful degradation; the whole run fails when the primary is unavailable.
- **`supportsDeveloperRole=false`** on non-native endpoints silently drops/reshapes a developer/system role — instructions may land differently than intended.
- **Streaming transport mismatch:** forcing `websocket` where a provider expects SSE (or vice-versa) can truncate or drop the stream; use `OPENCLAW_DEBUG_SSE` to inspect.
- **`extra_body` / per-model params** override defaults — a stale override can pin a bad `maxTokens`/reasoning setting for one model only.

## Sources

- `https://docs.openclaw.ai/concepts/model-providers`, `https://docs.openclaw.ai/concepts/model-failover` — API shapes, rate-limit-only retry, fallback chain.
- `https://docs.openclaw.ai/gateway/config-tools` — transport, block streaming, per-model params.

> Note: ClawRouter routing internals are not documented; treat as opaque.

# Hermes — Error taxonomy (harness-introduced failure modes)

**At a glance:** Keystone file for trace attribution. Hermes classifies every API failure into a **23-member `FailoverReason` enum** with an explicit recovery policy (retry / rotate credential / compress / fail-fast), applies a jittered backoff (`base=5s, max=120s`, default **3 retries**), and enforces iteration (90/50) and stale-stream caps. The tables below map trace symptoms and error classes to their harness cause, knob, and source.

## `FailoverReason` enum — representative members

The full enum has 23 members; the most trace-relevant are:

| reason | trigger | policy |
|---|---|---|
| `rate_limit` | 429 / quota throttle | backoff then rotate credential |
| `overloaded` | 503/529 | backoff |
| `server_error` | 500/502 | retry |
| `timeout` | connection/read timeout | rebuild client + retry |
| `context_overflow` | context too large | **compress, not failover** |
| `content_policy_blocked` | safety filter | **deterministic, don't retry unchanged** |
| `auth_permanent` | auth fails after refresh | **abort (not retryable)** |
| `ssl_cert_verification` | TLS verify failure | **fail fast, not retryable** |
| `model_not_found` | 404 / invalid model | fall back to different model |
| `unknown` | unclassifiable | retry with backoff |

Additional members cover: `auth`, `billing`, `upstream_rate_limit`, `payload_too_large`, `image_too_large`, `provider_policy_blocked`, `format_error`, `invalid_encrypted_content`, `multimodal_tool_content_unsupported`, `thinking_signature`, `long_context_tier`, `oauth_long_context_beta_forbidden`, `llama_cpp_grammar_pattern`. See `agent/error_classifier.py:24-72` for the full enum and policies.

Each classified error carries `retryable`, `should_compress`, `should_rotate_credential`, and `should_fallback` hints used by the retry machinery.

## Retry & rate-limit machinery

- **Backoff:** jittered, `base_delay=5.0 s, max_delay=120.0 s`. **Default `api_max_retries = 3`**. Z.AI GLM overload gets an adaptive long tier with escalating delays `(30,60,90,120)` s.
- **Per-attempt one-shot guards:** prevent double-retry on auth refresh, thinking-signature mismatch, image shrink, encrypted-content replay, and similar per-attempt repairs.
- **Cross-session rate breaker:** writes `~/.hermes/rate_limits/nous.json`, `default_cooldown=300.0 s`; separates genuine account-limit 429s from upstream-capacity 429s to avoid false trips.

## Failure-mode table (symptom → cause → knob → source)

| # | Symptom in trace | Likely harness cause | Knob / control | Source |
|---|---|---|---|---|
| 1 | Turn ends abruptly, no natural answer, "summary"-like final | **Iteration-cap truncation** (90 parent / 50 subagent) + toolless summary call | `max_iterations=90`, `delegation.max_iterations=50` | `iteration_budget.py`, `turn_finalizer.py` |
| 2 | "Invalid API response after 3 retries" then fallback/give-up | **Retry exhaustion** on invalid responses | `api_max_retries=3` | `conversation_loop.py` |
| 3 | Reasoning stream cut mid-think (`BrokenPipeError`/`RemoteProtocolError`) | **Stale-timeout kill** of a long reasoning stream | `HERMES_STREAM_STALE_TIMEOUT=180`, per-model floors | `reasoning_timeouts.py`, `chat_completion_helpers.py` |
| 4 | Empty/blank model response | **Empty stream** (provider closed stream with no content) | — | `errors.py` |
| 5 | All Nous models blocked for minutes | **Cross-session rate breaker** (possibly false-trip on transient upstream 429) | `nous.json` `default_cooldown=300s` | `nous_rate_guard.py` |
| 6 | Hard stop after repeated identical failing tool calls | **Guardrail halt** | `tool_guardrails` `same_tool_failure_halt_after=8`, `hard_stop_enabled` | `tool_guardrails.py` |
| 7 | Tool result `status="timeout"` | **Tool batch timeout** (420 s) | `HERMES_CONCURRENT_TOOL_TIMEOUT_S=420` | `tool_executor.py` |
| 8 | Request 400s before generation | **Provider schema rejection** (bad tool/MCP schema) or `format_error` | Gemini/Moonshot sanitizers | `error_classifier.py`, `gemini_schema.py`, `moonshot_schema.py` |
| 9 | "Cannot compress further" / context reset | **Context-overflow → compression** that can't make >5% progress | `context.engine`, `threshold_percent=0.75` | `turn_context.py` |
| 10 | Tool ran with empty `{}` args | **Silent outbound arg repair** fallback | (repair path) | `message_sanitization.py` |
| 11 | Duplicated/dropped text after a length cutoff | **Length/stream-drop continuation nudge** | `finish_reason="length"` handling | `conversation_loop.py` |
| 12 | Hard fail, no retry, on auth/TLS/policy | **Non-retryable fail-fast** classes | `ssl_cert_verification`, `content_policy_blocked`, `auth_permanent` | `error_classifier.py` |
| 13 | Total work exceeds 90 iterations | **Subagents each have independent 50-cap** | `delegation.max_iterations=50` | `iteration_budget.py` |

## How to use this file

Start from the trace symptom (or the provider error's `FailoverReason`), confirm the mechanism, then check the knob in `~/.hermes/config.yaml` / env. Cross-link: rows 1/13 → [architecture-and-loop.md](architecture-and-loop.md); rows 3–5 → [model-interaction.md](model-interaction.md); rows 6–8/10 → [tool-calling.md](tool-calling.md) and [parsing-and-output.md](parsing-and-output.md); row 9 → [prompt-and-context.md](prompt-and-context.md).

## Sources

- `nousresearch/hermes-agent` `agent/error_classifier.py` — `FailoverReason` enum (23 members), classifier funnel.
- `nousresearch/hermes-agent` `agent/retry_utils.py`, `agent/nous_rate_guard.py` — backoff, `api_max_retries=3`, cross-session rate breaker.

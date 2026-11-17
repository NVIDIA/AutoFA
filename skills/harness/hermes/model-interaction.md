# Hermes — Model interaction

**At a glance:** Providers are configuration objects; `AIAgent` dispatches on `api_mode` across five wire formats (`chat_completions`, `anthropic_messages`, `bedrock_converse`, `codex_responses`, native Gemini/Vertex). Streaming assembles tool-call deltas by **assigning** the name and **concatenating** arguments. A set of **stale-stream timeouts with per-reasoning-model floors** governs when a live stream is killed, and failover walks a configured fallback chain.

## How it works

**Provider dispatch.** Provider configs carry `api_mode`, endpoints, auth, and vision flags. `AIAgent` dispatches on `self.api_mode`:
- `"chat_completions"` — default OpenAI-wire (streaming + non-streaming), also OpenRouter/most aggregators.
- `"anthropic_messages"` — native Anthropic `/v1/messages`; `tool_use`/`tool_result` blocks; OAuth (Claude Code creds + refresh); betas incl. 1M context.
- `"bedrock_converse"` — AWS Bedrock (boto3, its own timeouts).
- `"codex_responses"` — OpenAI Responses API / Codex.
- Native Gemini (`agent/gemini_native_adapter.py`), Vertex (`agent/vertex_adapter.py`).

**Streaming.** Tool-call assembly **assigns** `function.name` (atomic, first chunk) and **concatenates** `function.arguments` deltas.

**Failover.** A fallback chain is walked on failure, with model-identity rewrite and in-flight system-message sync to prevent a stale `Model:/Provider:` line shipping after failover. Fallback-exhausted cooldown is 5 s.

**Key defaults.** `max_tokens` from config `model.max_tokens`, fallback **4096** on the ephemeral path. Temperature can be omitted via `OMIT_TEMPERATURE` sentinel. **Stale/timeout defaults:** stream-stale `HERMES_STREAM_STALE_TIMEOUT`=**180 s**, non-stream stale `HERMES_API_CALL_STALE_TIMEOUT`=**90 s**, request `HERMES_API_TIMEOUT`=**1800 s**, native Anthropic **900 s**, `HERMES_STREAM_STALE_GIVEUP`=**5** consecutive stalls. Reasoning models get elevated stale-timeout floors (e.g. 600 s for deepseek-r1, o1/o3, nemotron-ultra; 240–300 s for claude-opus-4, o3-mini, qwen3) — see `agent/reasoning_timeouts.py` for the full table.

## Where errors can originate

- **Stale-stream kill of a live reasoning stream** mid-think (surfaces as `BrokenPipeError`/`RemoteProtocolError`); floors mitigate but the kill itself is harness-introduced.
- **Empty stream:** provider closes the stream with no response → `EmptyStreamError`.
- **Cross-turn stale circuit breaker:** repeated stale calls trip a giveup after 5 consecutive stalls.
- **Tool-name concatenation bug class:** some providers resend the full function name each chunk; Hermes *assigns* (not `+=`) to avoid `read_fileread_file` — a provider that behaves differently can still corrupt the name.
- **Failover identity drift:** without system-message sync, a stale `Model:/Provider:` line ships after failover.

## Sources

- `nousresearch/hermes-agent` `agent/chat_completion_helpers.py` — call helpers, streaming assembly, failover, stale-stream logic.
- `nousresearch/hermes-agent` `agent/reasoning_timeouts.py` — stale-timeout defaults + per-model floors.

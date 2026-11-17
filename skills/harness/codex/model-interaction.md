# Codex — Model interaction

**At a glance:** Responses request retry, streaming recovery, transport fallback,
context recovery, and server model rerouting are distinct mechanisms.

## How it works

The model client constructs a Responses request from instructions, input items,
tools, reasoning/output controls, and optional final-output schema. The inspected
provider configuration accepts `wire_api="responses"`; legacy `"chat"` is explicitly
rejected. A compatible endpoint must implement the expected wire contract, not
merely expose a model with the same name.

Provider defaults are `request_max_retries=4`, `stream_max_retries=5`,
`stream_idle_timeout_ms=300000`, and WebSocket connection timeout 15,000 ms.
Configured retry values are capped at 100 in the provider-info layer. These are
separate controls, not a maximum number of agent steps or a total wall-clock bound.

`handle_response_stream_error` uses typed retry classification and server retry
advice. On stream-budget exhaustion, an eligible client can switch from WebSockets
to HTTPS and reset the stream retry counter. This is transport fallback, not proof
of changing the model. Release builds can suppress the first WebSocket reconnect
notification, so visible notifications do not enumerate every attempt.

An enabled `UnboundedConnectionRetries` feature has a separate path for qualifying
sampling connection failures, with increasing waits from 5 to 60 seconds and
source/provider exclusions. Do not apply the ordinary retry ceiling to that path
without checking feature state. Context-window and usage-limit errors are handled
separately by the sampling/turn layer.

## Where errors can originate

Capture actual provider/base URL, authentication mode without secrets, model,
transport, request IDs when available, typed error, retry advice, and timing.
Distinguish failed connection establishment from an interrupted response after
partial items or tool activity. Count attempts from transport evidence, not UI
messages alone. Use explicit model-rerouting events to establish a model change;
do not infer it from a reconnection message or transport switch.

## Sources

- [Model client](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/client.rs),
  [provider defaults and wire API](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/model-provider-info/src/lib.rs).
- [Stream retries and fallback](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/responses_retry.rs),
  [sampling error boundaries](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/session/turn.rs).

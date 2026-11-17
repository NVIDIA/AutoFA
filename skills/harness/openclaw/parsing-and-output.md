# Openclaw — Parsing & output

**At a glance:** Model output is streamed as deltas routed to `assistant` / `tool` / `lifecycle` streams; **block streaming** emits completed assistant blocks as they finish. Because tool calls are **native function-calling**, malformed-argument handling is largely **delegated to the provider/runtime** rather than a custom text parser. On a terminal state with nothing renderable and a failed tool, a **fallback tool-error reply** is emitted.

## How it works

- **Streaming assembly** (`/concepts/agent-loop`, `/concepts/agent`): deltas are routed to the `assistant` / `tool` / `lifecycle` streams. "Block streaming sends completed assistant blocks as soon as they finish" (`blockStreamingDefault` off by default).
- **Native tool-call parsing:** since tool calls use provider function-calling, the parsing/validation of tool arguments is handled by the provider SDK / runtime, not a bespoke text/XML parser (contrast with harnesses that parse tool calls out of free text).
- **Fallbacks on terminal state** (`/concepts/agent-loop`):
  - If the run reaches terminal state with **no renderable payloads left and a tool errored**, Openclaw emits a **fallback tool-error reply** (unless a messaging tool already replied).
  - `agentCommand` emits a **fallback lifecycle end/error** if the embedded loop didn't produce one.

## Where errors can originate

- **Fallback replies can mask the real failure:** a synthesized tool-error reply or lifecycle end/error means the trace shows a "clean" terminal event even though the underlying turn produced nothing usable — the true cause (tool crash, empty model output) is one layer down.
- **Malformed model output** is delegated to the provider/runtime, so behavior varies by provider; there is **no documented harness-level repair/reprompt** for malformed output.
- **Block-streaming boundary** (`blockStreamingBreak` = `text_end` vs `message_end`): the wrong break point can split or delay a block, changing what an observer/trace captures mid-stream.

## Sources

- `https://docs.openclaw.ai/concepts/agent-loop`, `https://docs.openclaw.ai/concepts/agent` — delta routing, block streaming, fallback replies.

> Gaps (not documented): explicit behavior on malformed model output (parse/repair/reprompt), and the exact serialization of tool results back into the next model message.

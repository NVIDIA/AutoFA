# Hermes — Parsing & output

**At a glance:** Output handling is defensive and harness-side: a streaming scrubber strips `<think>`-family tags across delta boundaries, surrogate/control-char sanitizers prevent `json.dumps` crashes, and an **outbound tool-arg repair** coerces malformed JSON (bounded to 50 iterations, falling back to `"{}"`). Error-body reads are bounded to 64 KB / 10 s so diagnostics never hang or raise.

## How it works

- **Visible-text extraction:** handles str / list-of-parts / Responses shapes, skipping image/audio parts.
- **Streaming reasoning scrub:** a per-delta state machine strips `<think>/<thinking>/<reasoning>/<thought>/<REASONING_SCRATCHPAD>` (case-insensitive), holding back partial tags across deltas; resets each turn; on flush, an unterminated block's content is discarded (leak-avoidance).
- **Surrogate & control-char sanitation:** lone surrogates → U+FFFD; raw control chars inside JSON strings are escaped; walks content/name/tool_calls/reasoning fields.
- **Outbound tool-arg repair:** tries `json.loads(strict=False)` + reserialize, then strips trailing commas, closes/removes unbalanced braces/brackets (bounded to **50** iterations), falls back to `"{}"` rather than crash.
- **Bounded error-body reads:** capped at **64 KB** with a hard **10 s** deadline on a daemon worker thread; **never raises**.

## Where errors can originate

All of these are **harness-side** transformations that can silently change semantics:
- **Reasoning leakage:** per-delta scrubbing can leak an opening `<think>` if state isn't preserved across deltas; conversely an unterminated block **truncates the answer** on flush.
- **Silent arg rewriting:** repair can coerce malformed args to `"{}"`, so a tool may run **with no arguments** (logged as a WARNING).
- **`json.dumps` crash inside the OpenAI SDK:** lone surrogates from byte-level reasoning models crash serialization unless sanitized first.
- **Truncated diagnostics:** the bounded error-body read may return only partial provider error text on stall/oversize — a real error can look vaguer than it is.

## Sources

- `nousresearch/hermes-agent` `agent/think_scrubber.py` — streaming reasoning scrub, tag set, flush behavior.
- `nousresearch/hermes-agent` `agent/message_sanitization.py`, `agent/bounded_response.py` — surrogate/control sanitation, outbound tool-arg repair, bounded error read.

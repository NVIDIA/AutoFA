# Pi — Parsing and output

**At a glance:** Provider streams normalize into assistant content blocks. Partial
arguments, completed calls, validated execution input, and final text have different
contracts; streaming previews are not authoritative completed output.

## How it works

`parseStreamingJson` attempts normal/repair parsing, then partial JSON parsing, and
can return `{}` while data is incomplete. String repair handles raw control characters
and invalid backslash escapes. This helper does not prove that every provider uses
the same final parser or that an empty preview was the executed argument object.

Tool preparation runs input normalization and `validateToolArguments`, which can
coerce values according to the schema before validation. The edit tool additionally
normalizes supported argument shapes. Preserve the provider's original arguments
and the post-normalization input when investigating changed semantics.

The loop rejects execution of tool calls from a `length`-terminated assistant
message. Tool validation errors and exceptions become `toolResult` feedback with
`isError`; ordinary text in the response is not a tool call or execution receipt.

JSON/RPC wire `message_update` events omit the SDK's cumulative `message` and
`partial` snapshots. They carry deltas indexed by `contentIndex`; completed block
events and `message_end.message` are authoritative. Appending each SDK snapshot
as if it were a delta duplicates content, while expecting snapshots on the wire
drops it. The `usage` field is cumulative, not an increment to sum per update.

Print mode's final text is a presentation surface, not a schema-validated evaluator
result. A wrapper requesting JSON or a custom completion tool must establish its
own validation contract from source and artifacts.

## Where errors can originate

Separate malformed provider arguments, permissive partial parsing, schema coercion,
tool rejection, and output serialization. Use terminal messages rather than treating
a partially streamed tool block as runnable. Check whether a downstream collector
mistook retry completion or one low-level `agent_end` for full session completion.

## Sources

- [JSON helpers](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/ai/src/utils/json-parse.ts),
  [argument validation/coercion](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/ai/src/utils/validation.ts).
- [Wire event conversion](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/modes/json-event.ts),
  [tool execution boundaries](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/agent/src/agent-loop.ts).

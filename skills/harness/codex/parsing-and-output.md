# Codex — Parsing and output

**At a glance:** The harness interprets structured Responses items and tool-specific
payloads. Plain final text, function arguments, custom patch syntax, and JSON-schema
output are separate parsing contracts.

## How it works

- Function calls carry serialized arguments; custom-tool calls carry an input string.
  The router preserves name/namespace and `call_id`, then delegates parsing to the
  registered handler. Client-side tool search has its own parsed argument type.
- `apply_patch` has a patch grammar and application/verification path. Its input
  must not be evaluated as ordinary JSON merely because other tools use JSON
  schemas. Shell/unified-exec calls deserialize their own argument structures.
- Recoverable `FunctionCallError` values become tool feedback; fatal errors stop
  the request/turn path. Do not assume a universal repair/re-prompt algorithm across
  all handlers or equate a parser failure with a command execution failure.
- `--output-schema` supplies a schema for the final response via the model request.
  It is separate from tool argument validation. The CLI's `--json` selects the
  event-stream format; it does not by itself request a JSON final answer.
- `--output-last-message` writes the final message, which omits the intervening
  trajectory. A consumer still needs to parse/validate its result and inspect the
  terminal turn outcome.

App Server deltas and item-completed payloads, exec item updates, and rollout
response items expose different projections. Reconstruct by item/call identity;
do not append a completed aggregate as if it were another text delta.

## Where errors can originate

Preserve raw call input, parser feedback, normalized output, and consumer behavior.
An `item.completed` event marks a terminal lifecycle state and can contain failed
command or patch status. `turn/completed` in App Server can carry `interrupted` or
`failed`; inspect the status field. Neither valid structured output nor normal
turn completion establishes that a code change passed the evaluator.

## Sources

- [Tool routing](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/tools/router.rs),
  [patch handler](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/tools/handlers/apply_patch.rs),
  [tool error feedback](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/tools/parallel.rs).
- [Final-schema request construction](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/client.rs),
  [official exec output guide](https://learn.chatgpt.com/docs/non-interactive-mode),
  [App Server events](https://learn.chatgpt.com/docs/app-server).

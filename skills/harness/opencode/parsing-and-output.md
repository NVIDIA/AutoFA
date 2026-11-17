# OpenCode — Parsing and output

**At a glance:** Stream events become typed session parts; plain text, reasoning,
tool calls, structured completion, and transport errors have separate handling.
The terminal rendering is only one projection.

## How it works

- The AI SDK adapter maps deltas and finished items to common `LLMEvent` values.
  Unsupported finish reasons become `unknown`; a raw provider `network_error`
  finish is raised as a stream error rather than treated as normal completion.
- The processor assembles text/reasoning parts and tool lifecycle records. A
  completed text hook may transform text before persistence. Unfinished tool calls
  are marked `error` with interruption metadata during cleanup.
- In the AI SDK path, `experimental_repairToolCall` first tries a lowercased tool
  name if it matches an available tool. Otherwise it redirects to the internal
  `invalid` tool with the original name and validation error. This is not a general
  JSON repair algorithm, nor evidence that the requested tool executed.
- For a user format of `json_schema`, the loop installs `StructuredOutput` and
  requires a tool call. A captured result is stored as `assistant.structured` and
  ends the loop. A terminal response without that result becomes
  `StructuredOutputError`; the observed branch records `retries: 0`.
- Provider content filtering has an explicit error path. A final text part, a
  step finish, and a successful structured result should not be conflated.

## Where errors can originate

Compare raw provider data when available, normalized calls, schema errors, executed
calls, and the result exported to the evaluator. Case normalization or routing to
`invalid` can explain why exported tool names differ from model output. Structured
output validates a contract; it does not verify a repository patch or benchmark
answer. A serializer or wrapper dropping `structured` can independently lose the
result even after the harness captured it.

## Sources

- [AI SDK event adapter](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/llm/ai-sdk.ts),
  [tool-call repair](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/llm.ts).
- [Structured-output handling](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/prompt.ts),
  [part persistence and cleanup](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/processor.ts).

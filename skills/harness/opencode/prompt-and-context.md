# OpenCode — Prompt and context

**At a glance:** Persisted history, the model request, and the compacted continuation
are not identical. Prompt hooks, provider transforms, tool-output truncation, and
compaction can each change what the model sees.

## How it works

Request preparation combines the agent prompt (or provider-specific default),
environment context, discovered instructions, skills/MCP instructions, and any
user-supplied system text. Instruction discovery considers global/project
`AGENTS.md`, compatibility fallbacks, and configured file/URL `instructions`.
Inspect discovered paths and effective content; do not assume every ancestor or
every compatibility file was concatenated.

Plugins can transform system text and chat messages before provider conversion.
`MessageV2.toModelMessagesEffect` and provider transforms normalize the stored parts
to the selected model's wire format. A session export is therefore not a complete
capture of the actual HTTP request.

Automatic compaction is enabled unless `compaction.auto` is false. The overflow
calculation uses model limits, observed usage, and output/reserved capacity, rather
than a universal percentage. Compaction produces a summary and retains recent
history within a token budget. `preserve_recent_tokens` overrides an inspected
default of 25% of usable input, clamped to 2,000–15,000 tokens; `tail_turns` can
further restrict retained turns. Plugins can replace the compaction prompt.

Old tool-output pruning is separate and **disabled by default at this revision**.
When enabled, it protects roughly 40,000 recent tool-output tokens and prunes only
when more than 20,000 older tokens can be cleared, excluding protected skill output.
Model-facing conversion substitutes cleared output; do not equate that with proof
the original execution never occurred.

## Where errors can originate

For a lost constraint, compare the original task, instructions, transformed request,
summary, and retained tail. Separate failure to retain a fact from failure to use a
fact that remained visible. Check [tool truncation](tool-calling.md) before blaming
compaction. Disabling auto-compaction does not increase the provider context window.

## Sources

- [Instruction discovery](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/instruction.ts),
  [request assembly](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/llm/request.ts).
- [Compaction and pruning](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/compaction.ts),
  [overflow calculation](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/overflow.ts).

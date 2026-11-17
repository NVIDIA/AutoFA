# OpenCode — Architecture and loop

**At a glance:** The session loop coordinates model steps, pending subtasks, and
compaction. Provider request retries live inside the stream processor and are not
additional user turns.

## How it works

1. `SessionPrompt.run` loads history after compaction filtering and locates the latest
   user, assistant, completed assistant, and pending tasks.
2. Pending subtasks or compaction are processed before the next ordinary model call.
   The selected agent supplies permissions and optional `steps`.
3. The loop resolves tools, runs message hooks, builds instructions and model history,
   and asks `SessionProcessor.process` to consume the stream.
4. Tool results normally cause another model call. A terminal assistant finish for
   the latest user exits only when there are no relevant outstanding tool calls;
   even provider `stop` can require continuation when tool calls were emitted.
5. The processor returns `continue`, `compact`, or `stop`. Errors, blocked tool
   execution, interruption, and successful structured output have distinct paths.

`agent.steps` defaults to no finite cap (`Infinity`). At the configured threshold,
the inspected loop appends `MAX_STEPS_PROMPT`, requesting a text-only summary.
**The same call still receives the resolved tools; this code path is an instruction,
not an unconditional numeric loop break or removal of tools.** Verify actual
termination rather than inferring a hard action budget from the setting.

The `task` tool associates child sessions through `parentID`, supports continuation
with `task_id`, and checks `subagent_depth` (default `1`). Child permission derivation
and result forwarding need separate inspection from the parent's final response.

## Where errors can originate

- Repeated similar calls are not automatically a `doom_loop` event. The processor
  checks three recent parts for the same tool and serialized input, then asks the
  permission system. This is not a general detector of semantically repetitive work.
- An outer task timeout can stop an otherwise continuing session. Compare evaluator
  deadlines with retry waits, shell deadlines, and child execution.
- A child completion does not establish that the parent integrated or verified its work.

## Sources

- [Loop and structured completion](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/prompt.ts),
  [processor and loop guard](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/processor.ts).
- [Step-limit instruction](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/core/src/session/runner/max-steps.ts),
  [child-session execution](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/tool/task.ts).

# OpenCode — Error taxonomy

**At a glance:** These categories guide investigation; they are not an upstream
error enum or ground-truth labels. Confirm the first failing boundary and subsequent
recovery before proposing a harness change.

## How to use this file

| Candidate category | Observable symptom | Confirm before attribution | Control/source |
|---|---|---|---|
| Tool availability or permission | Missing schema, pending approval, denied call | Actual agent, merged rules, plugins, and approval outcome | `permission`, agent selection; [tools](tool-calling.md) |
| Invalid call | Schema feedback or `invalid` result | Original name/arguments, normalization, and whether execution started | `Tool.wrap`, `experimental_repairToolCall` |
| Command/environment failure | Nonzero exit, timeout, missing file/dependency | Working directory, `metadata.exit`, output, and later recovery | Shell execution and per-call timeout |
| Repetition guard | `doom_loop` approval | Three qualifying recent parts and identical serialized inputs | `SessionProcessor`, `doom_loop` permission |
| Step-limit mismatch | Work continues past configured `steps` or ends with a summary | Injected limit instruction, actual calls, outer deadline | `SessionPrompt`; [loop](architecture-and-loop.md) |
| Provider retry exhaustion | Retry status then session error | Error classification, attempts, endpoint, and headers | `SessionRetry`; [model interaction](model-interaction.md) |
| Context or observation loss | Needed fact absent in later call | Output preview/full file, pruning, compaction summary, retained history | `tool_output`, `compaction`; [context](prompt-and-context.md) |
| Invalid structured completion | Text-only result or `StructuredOutputError` | Requested schema, completion tool call, stored `structured` field | [Parsing](parsing-and-output.md) |
| Child/integration failure | Task result missing, child failure, depth rejection | `parentID`, `task_id`, child history, parent consumption | `subagent_depth`, `TaskTool` |
| Interrupted execution | Aborted tool part or incomplete stream | Cancellation timing, cleanup metadata, outer runner status | Processor cleanup and client/wrapper |
| Export visibility gap | Missing reasoning, children, or input/output content | CLI flags, parent filtering, sanitization, source session | [Trace format](trace-format.md) |
| Unsupported success claim | Final prose disagrees with artifacts/verifier | Observed tests and evaluator output, not just session completion | External evaluator and tool evidence |

## Where errors can originate

Report session/message/part/call IDs, exact feedback, evaluated commit and settings,
the mechanism, plausible alternatives, and a focused verification step. Missing
evidence is not a negative observation. Permission enforcement can be correct even
when it prevents the task; a model continuing after accurate error feedback can be
a model issue without an implementation defect. Keep recovered local errors
separate from the causal task failure.

## Sources

- [Processor errors and lifecycle](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/processor.ts),
  [loop decisions](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/prompt.ts).
- [Child execution](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/tool/task.ts),
  [retry policy](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/retry.ts).

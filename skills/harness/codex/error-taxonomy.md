# Codex — Error taxonomy

**At a glance:** These are investigation categories, not a Codex error enum.
Establish a mechanism from the evaluated revision and trace before assigning cause.

## How to use this file

| Candidate category | Observable symptom | Confirm before attribution | Control/source |
|---|---|---|---|
| Tool-contract mismatch | Unknown tool, bad arguments, patch parse failure | Actual schema, payload type, name/namespace, parser feedback | [Tools](tool-calling.md), [parsing](parsing-and-output.md) |
| Permission or sandbox boundary | Declined/rejected call, inaccessible path/network | Effective profile, approval decision, first attempt and eligible retry | `ToolOrchestrator`, exec policy |
| Command/environment failure | Nonzero exit, timeout, missing dependency | Actual cwd/environment, output, terminal exit and recovery | Unified exec/selected runtime |
| Unconsumed process result | Agent reports completion while command remains active | Returned `session_id`, later polling, process terminal state | `exec_command`, `write_stdin` |
| Stream/retry failure | Reconnection messages then turn failure | Request vs stream attempts, transport fallback, typed error, feature flags | [Model interaction](model-interaction.md) |
| Context/instruction loss | Required detail absent from later request | Trust/discovery, doc budget, tool truncation, compaction replacement | [Context](prompt-and-context.md) |
| Unexpected continuation | Extra requests after apparent final response | Pending input, Stop hook feedback, context operations, child events | [Loop](architecture-and-loop.md) |
| Child integration failure | Child result exists but parent work is incomplete | Child identity/history, shared edits, parent consumption and checks | Child-agent tools and application |
| Structured-output mismatch | Consumer receives prose, malformed JSON, or wrong fields | `--output-schema`, final text, turn status, consumer validation | Client request/output handling |
| Event/projection mistake | Collector marks failed item successful or duplicates text | Event family, terminal status, item IDs, delta/aggregate distinction | [Trace format](trace-format.md) |
| Artifact visibility gap | No rollout or missing tool/child details | Ephemeral mode, history retention, compressed/linked files, wrapper export | Rollout recorder and persistence policy |
| Unsupported success claim | Normal turn ends but verifier fails | Executed checks and evaluator artifacts | External task/evaluator boundary |

## Where errors can originate

Keep recovered local errors separate from terminal failure. A permitted retry,
correct policy denial, or model ignoring accurate feedback does not independently
establish a runtime defect. Report task/repeat, thread/turn/item/call IDs, exact
feedback, revision/configuration, mechanism, alternatives, and a concrete validation
step. If only final text or a filtered event stream is available, disclose the
missing evidence instead of reconstructing unobserved actions.

## Sources

- [Turn and sampling decisions](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/session/turn.rs),
  [tool orchestration](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/tools/orchestrator.rs).
- [Exec statuses](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/exec/src/exec_events.rs),
  [rollout retention policy](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/rollout/src/policy.rs).

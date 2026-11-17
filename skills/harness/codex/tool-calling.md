# Codex — Tool calling

**At a glance:** Tool calls carry native IDs and typed payloads. Availability,
argument parsing, approval, sandbox selection, execution, and output truncation
are separate stages.

## How it works

`ToolRouter::build_tool_call` maps function-call arguments, custom-tool input, and
eligible client-side tool-search calls to internal dispatch. Model/features and
registered handlers determine which schemas exist. Inspect the actual tool list;
shell, unified exec, apply-patch, MCP, search, and code-mode interfaces are not
interchangeable contracts.

`ToolOrchestrator` coordinates execution approval, sandbox strategy, and eligible
retry after sandbox denial. Escalation depends on policy and runtime checks; it is
not unconditional fallback. MCP services execute in their own environment and do
not inherit proof of local filesystem isolation merely from Codex's shell sandbox.

| Boundary | Evidence to retain | Interpretation |
|---|---|---|
| Function/custom call | `call_id`, name/namespace, original arguments/input | Correlate with the matching output; prose is not dispatch |
| Unified exec | `cmd`, workdir/environment, shell, `yield_time_ms`, optional `timeout_ms` | Yield duration and termination deadline differ |
| Running process | Returned `session_id`, subsequent `write_stdin` results | A yielded call is not a completed command |
| Terminal command | Exit code, output, cancellation/timeout status | Empty output does not establish success |
| Patch | Submitted patch, parse/approval outcome, applied file changes | Patch intent alone is insufficient |
| MCP | Server/tool, arguments, result/error, approval outcome | Separate server-side failure from model input failure |

The inspected unified-exec default initial yield is 10 seconds. Interactive and
one-shot execution lifetimes differ; the one-shot path applies a deadline. Output
has its own token/byte limits, including a default unified-exec output allowance of
10,000 tokens. These are not a global model context or task timeout.

## Where errors can originate

Determine whether a rejected call reached execution. Schema errors, policy denial,
sandbox denial, nonzero exit, and provider rejection require different remedies.
Check later process output before classifying a yielded command as unfinished.
Preserve truncated output evidence and avoid assuming a displayed preview contains
everything the process produced.

## Sources

- [Router](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/tools/router.rs),
  [approval/sandbox orchestration](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/tools/orchestrator.rs).
- [Exec arguments](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/tools/handlers/unified_exec.rs),
  [lifetime selection](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/tools/handlers/unified_exec/exec_command.rs),
  [output representation](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/tools/context.rs),
  [approval/security guide](https://learn.chatgpt.com/docs/agent-approvals-security).

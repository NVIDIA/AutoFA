# Pi — Error taxonomy

**At a glance:** These are candidate mechanisms for investigation, not Pi error
codes or automatic labels. A failed local operation can recover without causing
the final task failure.

## How to use this file

| Candidate category | Observable symptom | Confirm before attribution | Control/source |
|---|---|---|---|
| Wrong resource/tool loadout | Missing tool, unexpected prompt, extension absent | Project trust, enabled resources, active/deferred tools, CLI overrides | [Configuration](configuration.md), resource loader |
| Invalid or transformed arguments | Validation feedback or different executed values | Raw arguments, normalization, schema coercion, result | [Parsing](parsing-and-output.md) |
| Truncated generation | `stopReason="length"`, failed tool batch | Output limit, rejected calls, compact-and-retry outcome | Agent loop and session recovery |
| Tool/environment failure | `isError`, nonzero shell exit, timeout, edit rejection | Correlated terminal result, cwd, output, subsequent recovery | [Tool contracts](tool-calling.md) |
| Parallel side-effect conflict | Calls overwrite/interfere with shared state | Batch timing, execution mode, tool-specific serialization, hooks | `Agent.toolExecution`, tool implementation |
| Premature collector stop | Trace ends at `agent_end` while work remains | `willRetry`, queues, compaction, eventual `agent_settled` | [Loop](architecture-and-loop.md), JSON/RPC consumer |
| Provider exhaustion | Retried stream failures or terminal limit error | Agent/provider counters, classification, route, timing | [Model interaction](model-interaction.md) |
| Context reconstruction error | Model appears to ignore raw-file content | Selected leaf, compaction boundary, `context_edit`, actual request | [Context](prompt-and-context.md) |
| Lost observation | Relevant output absent from prompt | Read head/bash tail truncation, full-output file, extension transform | Tool output limits |
| MCP exposure/integration | Connected tool unavailable or server error | Exposure/discovery, connection status, `isError`, parent call | MCP settings and extension pipeline |
| Trace accounting error | Duplicated text, inflated tokens, missing final response | Delta vs snapshot contract, cumulative usage, final message | [Trace format](trace-format.md) |
| Unsupported completion claim | Settled run reports success without verification | Tool evidence and external evaluator artifacts | Application/evaluator boundary |

## Where errors can originate

For each finding, retain task/repeat identity, session and branch, entry or tool-call
IDs, exact feedback, revision/settings, mechanism, alternatives, and a verification
step. Do not infer that every stored message was model-visible or that every
`tool_execution_start` reached execution. Distinguish default coding-agent behavior
from extension, SDK, experimental-runtime, and benchmark-wrapper behavior.

## Sources

- [Agent loop](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/agent/src/agent-loop.ts),
  [session recovery](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/agent-session.ts),
  [projection](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/session-manager.ts).

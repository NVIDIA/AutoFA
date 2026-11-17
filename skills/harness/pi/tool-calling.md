# Pi — Tool calling

**At a glance:** Tool calls use `{type:"toolCall", id, name, arguments}` blocks;
feedback is a `toolResult` message keyed by `toolCallId`, with `isError`. Installed,
active, deferred, and executed tools are distinct sets.

## How it works

The generic loop resolves tools, normalizes/validates arguments, applies before/after
hooks, executes, and emits terminal feedback. Unknown tools, validation failures,
blocked calls, and exceptions produce error results. `tool_execution_start` is
emitted before preparation, so that event alone does not prove an OS operation ran.

| Tool boundary | Inspected contract | Failure-analysis implication |
|---|---|---|
| `read` | Head of a text file, default 2,000 lines/50 KiB; `offset`/`limit` for continuation; images supported | Missing trailing content can be truncation |
| `bash` | Combined output tail at 2,000 lines/50 KiB; full-output path when truncated | Recover the saved file for omitted output |
| Shell timeout | Optional `timeout` in **seconds**, no default deadline | A wrapper may time out first; do not import OpenCode's millisecond contract |
| Shell nonzero exit | Error result plus `structuredContent.exit_code` and output | Inspect error status; scripts can receive a resolved structured result |
| `edit` | Targeted replacements in `edits[]`; each region unique and non-overlapping | An emitted edit is not proof of a write; inspect returned diff/error |
| Extensions/SDK tools | Can replace execution and modify input/output | Name alone does not establish built-in semantics |

MCP support includes stdio and streamable HTTP, with exposure modes controlling
direct visibility and discovery through code mode/tool search. MCP results preserve
`isError` and structured content; code-mode consumers receive the whole result.
Every MCP call passes through Pi's tool pipeline and its hooks. Discovering a tool
does not itself execute it, and a parent code-mode call can contain several calls.

## Where errors can originate

Correlate the model block, normalized arguments, hook decisions, terminal event,
and persisted result. Parallel calls can interact through files and external state.
Permission gates, isolation, and delegation supplied by extensions or wrappers
must be demonstrated from their configuration, not assumed from the Pi label.

## Sources

- [Dispatch and validation boundary](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/agent/src/agent-loop.ts),
  [shell tool](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/tools/bash.ts),
  [read tool](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/tools/read.ts).
- [Edit tool](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/tools/edit.ts),
  [MCP conversion](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/extensions/mcp/tools.ts),
  [MCP contracts](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/docs/mcp.md).

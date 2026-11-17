# Pi — Trace format

**At a glance:** Persisted session JSONL is a tree; print/JSON and RPC streams are
lifecycle events. They are different schemas even though both use JSON lines.

## How it works

| Surface | Identifiers and shape | Interpretation |
|---|---|---|
| Session header | `type:"session"`, `version:3`, `id`, `timestamp`, `cwd`, optional `parentSession` | File/session identity; no entry `parentId` required |
| Tree entry | `type`, `id`, `parentId`, ISO timestamp | Follow the selected leaf to reconstruct a branch |
| `message` entry | Nested `message` with role, content and message timestamp | Assistant metadata includes model/provider/API, usage, `stopReason` |
| Tool call/result | Assistant `toolCall.id`; result `toolCallId`, `toolName`, `isError` | Join execution to feedback |
| Context bookkeeping | `compaction`, `branch_summary`, `context_edit`, model/thinking changes | Changes continuation context without deleting raw history |
| Non-conversation data | `custom`, `usage`, labels/session info | Not automatically a model-visible user turn |
| JSON mode | `pi --mode json -p ...`: header followed by session events | Live projection, separate from persisted entries |
| RPC mode | JSONL commands/responses/events; command IDs correlate replies | No initial session header; query session state |

Default session files are grouped by encoded working directory under
`~/.pi/agent/sessions/`, with timestamp/session-ID filenames. Location overrides
and ephemeral mode matter when looking for artifacts.

Use `buildSessionProjection` for active context: apply branch ancestry, the newest
compaction's `firstKeptEntryId`, and latest target-specific context edits. Do not
concatenate all `message` rows across abandoned branches.

Live events include `message_start/update/end`, `tool_execution_start/update/end`,
`turn_end`, `agent_end`, retry/compaction events, and `agent_settled`. Final
`message_end.message` supersedes partial reconstruction. Wire updates are deltas;
in-process SDK updates also carry cumulative snapshots. Read stdout continuously
and split JSONL on LF, preserving Unicode separators inside valid strings.

## Where errors can originate

Capture task/repeat, session, branch leaf, entry IDs, and tool-call IDs. An
`agent_end` with recovery pending is not final completion; `agent_settled` is not
proof of evaluator success. Usage can include summaries/cache warming. Preserve
raw entries alongside the model projection so omitted failed attempts remain
auditable. Missing session files or provider payloads are evidence limitations.

## Sources

- [Official session format](https://pi.dev/docs/latest/session-format) and
  [CLI integration](https://pi.dev/docs/latest/cli-integration).
- [Session schema and reconstruction](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/session-manager.ts),
  [wire transformation](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/modes/json-event.ts),
  [event reference](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/docs/json.md).

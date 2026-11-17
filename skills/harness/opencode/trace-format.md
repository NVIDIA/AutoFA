# OpenCode — Trace format

**At a glance:** Distinguish persisted session data, server events, CLI JSONL, and
sanitized exports. None alone guarantees a full provider-request capture or a
complete child trajectory.

## How it works

| Surface | Shape and identifiers | Interpretation |
|---|---|---|
| Session export | `opencode export <sessionID>` writes JSON with `info` and `messages`; each message has `info` and `parts` | Richer session artifact than terminal text |
| CLI event stream | `opencode run --format json` emits JSONL with `type`, `timestamp`, `sessionID`, plus `part` or `error` | Selected events, not the raw provider stream |
| Tool part | `id`, `sessionID`, `messageID`, `callID`, `tool`, `state` | Join input to completed/error result by call ID |
| Assistant metadata | `parentID`, provider/model, finish/error, usage and cost | Connect model steps to their user message |
| Child session | Session `parentID`; task output carries child identity | Retrieve child's history separately |
| Persistence | SQLite session, message, and part tables | Storage schema depends on revision |

CLI event types include `step_start`, `step_finish`, `text`, `tool_use`, and `error`.
`tool_use` is emitted for terminal tool states, including errors; its name does not
mean the tool succeeded. Reasoning events depend on the CLI's thinking setting.
The inspected CLI filters message parts to the active parent session, so a child
tool transcript cannot be reconstructed from those parent events alone.

`opencode export <sessionID> --sanitize` replaces content in prompts, text, tool
inputs/results, attachments, paths, and other fields. Redaction placeholders are
not literal model observations. The export does not expand a truncated tool-output
file or supply all transient request transforms automatically.

## Where errors can originate

Preserve task/repeat identity alongside session/message/part/call IDs, original
timestamps, terminal states, and output paths. Check shell exit metadata even when
the part is completed. Use the exporter matching the evaluated version; do not
assume the legacy export command and newer CLI surfaces have identical schemas.
If provider requests, child sessions, or saved output are unavailable, state that
scope rather than inferring absence of behavior.

## Sources

- [CLI documentation](https://opencode.ai/docs/cli/),
  [JSONL projection](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/cli/cmd/run.ts),
  [export and redaction](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/cli/cmd/export.ts).
- [Session SQL schema](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/core/src/session/sql.ts),
  [message conversion](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/message-v2.ts).

# Codex — Trace format

**At a glance:** Identify the event family before interpreting fields. Exec JSONL,
App Server notifications, and persisted rollouts use different names and lifecycle
semantics.

## How it works

| Surface | Shape/identity | Interpretation |
|---|---|---|
| `codex exec --json` | `thread.started` with `thread_id`; `turn.started/completed/failed`; `item.started/updated/completed`; `error` | Automation projection, not raw Responses SSE |
| Exec item | `id`, type such as `command_execution`, `agent_message`, `file_change`, `mcp_tool_call` | Inspect type-specific status, `exit_code`, and output |
| App Server | Notifications such as `item/started`, `item/completed`, `item/agentMessage/delta`, `turn/completed` | Join thread/turn/item IDs; requests/approvals require their own replies |
| Persisted rollout | Timestamped records with `type`/`payload` and optional ordinal | Internal history and bookkeeping, not the exec schema |
| Rollout payload types | `session_meta`, `turn_context`, `response_item`, `event_msg`, `compacted`, plus revision-specific context/accounting records | Apply the matching history reconstruction rules |
| Model tool pair | Function/custom call and matching output keyed by `call_id` | Stronger execution evidence than assistant prose |

Rollouts normally live under `$CODEX_HOME/sessions/YYYY/MM/DD/` with
`rollout-<timestamp>-<thread-id>.jsonl` names. Archival, compression, revert lineage,
and history mode can change how artifacts are stored or linked. Use the matching
recorder/reader rather than assuming every session is one uncompressed linear file.
Ephemeral runs need separately captured client events.

Exec `item.completed` includes failed terminal items. App Server `turn/completed`
can carry `completed`, `interrupted`, or `failed` status. Read these values rather
than interpreting the event verb as success. Exec usage and item payloads are
aggregates; do not count repeated updates as independent work.

Persistence policy filters response/event types and varies with history mode.
Compaction changes active context while older stored events remain evidence of
past execution. Reasoning visibility depends on what the provider/client actually
exposes; absent hidden reasoning cannot support a behavioral claim.

## Where errors can originate

Retain task/repeat, binary revision, thread/turn/item/call IDs, terminal statuses,
tool output, and evaluator artifacts. Retrieve child histories and follow process
handles when the parent contains only summaries or yielded output. A final-message
file alone cannot prove which commands ran or why the turn stopped.

## Sources

- [Exec event types](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/exec/src/exec_events.rs),
  [App Server protocol guide](https://learn.chatgpt.com/docs/app-server).
- [Rollout types](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/history/src/lib.rs),
  [recorder](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/rollout/src/recorder.rs),
  [persistence policy](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/rollout/src/policy.rs).

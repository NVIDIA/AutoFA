# Pi — Architecture and loop

**At a glance:** A low-level run contains assistant turns and tool batches. The
coding session can continue after that run through retry, compaction recovery, or
queued work; `agent_end` alone is not the final session boundary.

## How it works

1. `Agent` starts `runAgentLoop` with prompt messages or continues existing context.
   The loop declares tool-loadout changes and emits agent/turn/message events.
2. Before generation, preparation hooks can update context, model, and thinking.
   Streaming yields an assistant message; `error` or `aborted` ends this low-level run.
3. Tool calls are validated and executed, then their results enter history. A
   `length` finish causes all tool calls in that response to receive error feedback
   without execution, because their arguments may be incomplete.
4. Tool results and queued steering messages can request another assistant turn.
   `finishTurn` can also end or explicitly continue. Follow-up messages are consumed
   after the loop would otherwise stop.
5. `AgentSession` persists results, manages automatic recovery, and emits
   `agent_settled` when no automatic continuation remains.

The inspected `Agent` defaults to **parallel** tool execution. A configured
`toolExecution="sequential"` or a tool declaring `executionMode="sequential"`
switches the batch to sequential execution. Results and side effects therefore
need not follow the display order. Steering is checked at turn boundaries in this
loop; do not apply older descriptions of interrupting each pending tool to it.

There is no fixed numeric iteration cap in this loop. Termination depends on
response/tool state, hooks, cancellation, queued work, and enclosing application
budgets. An extension can implement a different stopping policy.

## Where errors can originate

Do not stop consuming a trace at the first `agent_end`: retry or overflow recovery
can generate another run. Distinguish explicit cancellation from tool failure and
queue starvation. Before alleging a race, recover batch membership, execution
mode, tool hooks, and actual side-effect timing. A settled run still requires
external verification of task success.

## Sources

- [Loop, batch selection, and hooks](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/agent/src/agent-loop.ts),
  [Agent defaults](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/agent/src/agent.ts).
- [Session recovery and settling](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/agent-session.ts),
  [event contract](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/docs/json.md).

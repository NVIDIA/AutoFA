# Codex — Architecture and loop

**At a glance:** A turn loops over sampling requests and tool feedback. Input
steering, stop hooks, context limits, and cancellation can change the next action;
one API response does not necessarily finish a user turn.

## How it works

1. Turn preparation resolves step settings, initial context, tools, and any required
   pre-turn compaction. The runtime tracks cancellation and pending user input.
2. Sampling builds a prompt from current history and base instructions, then streams
   Responses events. Tool requests are routed, executed, and recorded with outputs.
3. `model_needs_follow_up` and pending input determine continuation. Tool work or
   newly accepted input can produce another request using updated history.
4. Context/window thresholds may trigger compaction or a feature-specific context
   reset. These operations have different request types from ordinary sampling.
5. When no follow-up remains, Stop hooks can request continuation with feedback.
   Otherwise the turn ends; cancellation and errors have distinct terminal paths.

Tool concurrency is capability-based. `ToolCallRuntime` uses a shared lock for
parallel-safe calls and an exclusive lock for calls requiring serialization. A
provider returning multiple calls is not proof they ran concurrently. Code mode
and child agents introduce additional nested work and separate lifecycle events.

There is no single universal iteration count to infer from this loop. Sampling
retry limits, tool deadlines, context/token-budget features, child constraints, and
an evaluator's wall-clock budget govern different boundaries.

## Where errors can originate

Distinguish a tool finishing, the model finishing a response, the user turn ending,
and the benchmark passing. A Stop hook can explain an extra request after apparent
completion. A running command may yield a process handle rather than an exit code;
the model must consume later results when needed. Child completion does not prove
the parent integrated the work. Inspect interruption/steering events before
attributing an abandoned action to a model decision.

## Sources

- [Turn loop and stop hooks](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/session/turn.rs),
  [parallel tool runtime](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/tools/parallel.rs).
- [Process lifecycle](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/unified_exec/mod.rs),
  [App Server turn lifecycle](https://learn.chatgpt.com/docs/app-server).

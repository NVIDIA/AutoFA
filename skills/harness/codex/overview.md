# Codex — Overview

**At a glance:** OpenAI Codex's open-source Rust runtime coordinates model requests,
tools, permissions, context, and persistent threads. The model named Codex and the
Codex harness are separate components; the client and deployment also matter.

## How it works

- This pack covers `openai/codex`, primarily `codex-rs/core`, the noninteractive
  `exec` interface, and App Server. It does not establish the implementation of
  every hosted Codex service or a benchmark wrapper calling the CLI.
- A thread contains turns; a user turn can involve multiple model sampling requests,
  tool calls, approval waits, context changes, and child-agent work. `run_turn`
  behavior is implemented in `core/src/session/turn.rs` at the inspected revision.
- The model client speaks the Responses wire protocol, with transport and provider
  capability selection. Tool routing accepts native function calls and custom-tool
  payloads; enabled features, model metadata, MCP, and extensions shape availability.
- App Server exposes the runtime through a bidirectional client protocol. `codex exec`
  exposes a smaller JSONL event schema for automation. Persisted rollout records
  have a third schema and their own retention/projection policy.

## Instructions for the analysis agent

Record the binary/source revision, invocation/client, thread and turn IDs, actual
model/provider, permission profile, workspace/environment, hooks/features, and
outer evaluator limits. Read [trace-format.md](trace-format.md), then
[error-taxonomy.md](error-taxonomy.md). Trace a tool through its terminal result and
the evaluator's evidence rather than treating the assistant's final message as a
success flag. These references grant no tools to AutoFA and do not override its
analysis focus, scoring, or evidence rules.

## Where errors can originate

Separate model output, client protocol, runtime dispatch, permission enforcement,
process/MCP execution, provider transport, and evaluator conversion. A permission
denial can be correct behavior, and an incomplete exported trace can conceal
successful recovery. Availability in this repository is not proof of enablement.

## Sources

Inspected on 2026-09-29 at `0b1b78a4f1694e2b9e393d385c7b82ca714ca08a`
(upstream `main`, not a release guarantee). All implementation links pin that
revision; match the evaluated binary before applying defaults or protocol details.

- [Official noninteractive guide](https://learn.chatgpt.com/docs/non-interactive-mode)
  and [App Server guide](https://learn.chatgpt.com/docs/app-server).
- [Turn implementation](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/session/turn.rs),
  [exec entrypoint](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/exec/src/lib.rs).

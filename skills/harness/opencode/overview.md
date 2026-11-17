# OpenCode — Overview

**At a glance:** OpenCode is Anomaly's TypeScript coding-agent application. Its
session loop serves terminal, server/SDK, and other clients. Identify the client,
agent, model route, and enabled plugins before attributing a failure.

## How it works

- This pack covers `anomalyco/opencode`, with its Bun-based CLI and shared session
  implementation. It does not describe the older, separately maintained Go project
  with the same name, or an evaluation wrapper invoking OpenCode.
- `SessionPrompt` assembles a request; `LLM` prepares provider inputs;
  `SessionProcessor` records streamed messages, executes the tool lifecycle, and
  decides whether to continue, compact, or stop.
- Primary agents such as `build` and `plan` have different prompts and permissions.
  The `task` tool starts or resumes child sessions. MCP servers and plugins can
  change the tools, messages, provider options, and resulting observations.
- The inspected implementation normally adapts AI SDK streams. An experimental
  native LLM runtime has its own capability check and fallback to the adapter.
  A model name alone does not identify the active transport or tool contract.

## Instructions for the analysis agent

Establish revision, client/command, effective agent and permissions, provider/model,
plugins, workspace, and outer evaluator limits. Start with [trace-format.md](trace-format.md),
then [error-taxonomy.md](error-taxonomy.md). Correlate task, session, message, part,
and tool-call IDs; distinguish parent results from child execution evidence.
Treat categories as hypotheses and verify the mechanism at the evaluated revision.
These references grant no tools to AutoFA and do not override its evidence rules.

## Where errors can originate

A stopped session is not an evaluator verdict. Tool permissions, provider errors,
context transformations, client cancellation, and benchmark timeouts are separate
boundaries. Missing content in a CLI projection or sanitized export is a visibility
limit until the underlying session artifacts are checked.

## Sources

Source inspected on 2026-09-29 at commit
`7945de208964a49300d7f770d1a71d078db9a4c4` (upstream `dev`, not a release guarantee).
All implementation links in this pack pin that revision; website documentation can
move independently. Recheck older runs against their own commit.

- [Official introduction](https://opencode.ai/docs/) and [agents](https://opencode.ai/docs/agents/).
- [Session loop](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/prompt.ts),
  [LLM runtime selection](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/llm.ts).

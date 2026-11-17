# Pi — Overview

**At a glance:** Pi is an extensible TypeScript coding agent with a shared session
layer across interactive, print/JSON, RPC, and SDK use. Its provider library, generic
agent loop, coding tools, and installed extensions are separate attribution layers.

## How it works

- This pack covers `earendil-works/pi`, specifically the standard coding-agent path
  created by `createAgentSession`. Older references use `badlogic/pi-mono` or
  `@mariozechner/*`; record the evaluated package and commit instead of assuming
  those versions have current behavior.
- `packages/ai` implements provider/message abstractions; `packages/agent` supplies
  `Agent` and the tool loop; `packages/coding-agent` adds persistence, context,
  compaction, retry, resources, and client interfaces.
- The default tools are `read`, `bash`, `edit`, and `write`. Extensions and SDK callers
  can change tools and their execution, inject messages, route models, and alter
  provider requests. The inspected coding agent includes MCP integration and optional
  code mode; historical claims that Pi has no MCP do not describe this revision.
- Additional experimental harness/runtime modules coexist in the repository. Their
  existence is not evidence that a standard CLI run used them. Follow the entrypoint
  through `sdk.ts` to the instantiated `AgentSession`.

## Instructions for the analysis agent

Record revision, entrypoint, model/provider/API, effective settings, enabled tools
and extensions, session/branch identity, and outer runner limits. Read
[trace-format.md](trace-format.md), then [error-taxonomy.md](error-taxonomy.md).
Reconstruct the active model context rather than concatenating all JSONL messages.
These notes guide AutoFA's analysis and do not grant Pi tools or supersede scoring
and evidence requirements.

## Where errors can originate

A raw agent event, coding-session completion, tool result, and evaluator verdict
have different meanings. Project trust and extension loading can change behavior
before a prompt runs. Ordinary tools use the Pi process's OS permissions; an
extension-provided execution backend must be inspected separately.

## Sources

Inspected on 2026-09-29 at `33e203354391ba26937cd1e8a210b4db10e82c66`
(package manifest `0.87.1`; a source snapshot, not proof of an installed release).
Implementation links pin this commit; website pages track current documentation.

- [How Pi works](https://pi.dev/docs/latest/how-pi-works).
- [SDK construction](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/sdk.ts),
  [session layer](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/agent-session.ts),
  [package manifest](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/package.json).

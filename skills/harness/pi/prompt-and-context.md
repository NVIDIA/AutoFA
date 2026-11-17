# Pi — Prompt and context

**At a glance:** Pi builds context from the active session branch, compaction
boundaries, and context edits. The complete append-only history can contain messages
that are no longer sent to the model.

## How it works

- Resource loading discovers context files in the agent directory and project
  ancestry. Candidates include `AGENTS.override.md`, `AGENTS.md`, and compatibility
  names. The override replaces the same directory's ordinary file, not every other
  directory's instructions.
- `SYSTEM.md` replaces the default prompt; `APPEND_SYSTEM.md` adds to it. Trusted
  project files take precedence over their agent-directory counterpart. Context-file
  discovery and project executable-resource trust have different rules.
- Tool instructions and skill descriptions participate in prompting. Full skills
  load on demand; templates expand input. Extensions can change context and provider
  payloads, so stored session messages need not equal a wire request.
- `buildSessionContext`/`buildSessionProjection` follow the selected leaf's parent
  chain, apply compaction, and apply the latest `context_edit` for each target.
  Omission or replacement changes the model projection while preserving raw entries.
- Automatic compaction normally starts when context usage exceeds the model window
  minus `reserveTokens` (default 16,384). It aims to preserve about 20,000 recent
  tokens and summarizes older context. Per-model overrides and extensions can alter
  the behavior. `firstKeptEntryId` marks the retained boundary.
- Context-overflow and recoverable length errors can cause one compact-and-retry
  recovery; successful responses crossing the threshold can compact without replay.
  Branch summaries and manual compaction are separate operations.

## Where errors can originate

Check the selected branch, summary, retained messages, context edits, and extension
transforms before asserting that a fact was visible. A failed attempt may remain in
the raw file but be durably omitted from subsequent model input. `custom` entries
and usage accounting are not ordinary model messages; `custom_message` entries can
be. User shell messages marked `excludeFromContext` are also excluded.

For loss after compaction, preserve the original constraint and the next actual
request. An inaccurate summary, tool-output truncation, a wrong branch, and ignoring
visible evidence are different candidate causes.

## Sources

- [Resource discovery](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/resource-loader.ts),
  [configuration rules](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/docs/configuration.md).
- [Session projection](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/session-manager.ts),
  [compaction algorithm](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/compaction/compaction.ts),
  [recovery policy](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/agent-session.ts).

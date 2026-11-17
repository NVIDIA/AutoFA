# Codex — Prompt and context

**At a glance:** The next request is built from effective instructions and projected
history, not a concatenation of the visible terminal transcript. Context replacement
and tool-output truncation can remove different kinds of information.

## How it works

Base instructions, user/developer instructions, environment and permission context,
skills, tools, and conversation items contribute to a request. The active model and
features determine exact rendering. Inspect the submitted request when available;
the presence of a file or installed skill does not establish that its content was
included in that turn.

At this revision, project instruction discovery walks from the recognized project
root to the current directory, preferring `AGENTS.override.md` over `AGENTS.md` and
configured fallbacks. The default project-document budget is 32 KiB. Root markers,
trust, selected execution environments, and readable paths affect discovery; an
untrusted active project skips project instructions in this loader. Host-provided
instructions are a separate contribution.

Automatic compaction is selected from provider capabilities and enabled features:
remote V2 compaction, local summarization, or token-budget context reset. The last
is not a summarization operation. Thresholds derive from model/configuration state;
there is no universal percentage to apply across all Codex runs.

Local summarization replaces active history with selected user context and a
summary. Pre-turn/manual and mid-turn paths differ in when initial context is
reinjected. Rollout compaction checkpoints preserve replacement-history information
for reconstruction. Post-turn compaction can also run when configured.

## Where errors can originate

For a lost requirement, compare its original location, discovered instructions,
request before compaction, replacement context, and request afterward. A fact in an
old rollout record can be absent from the active model context. Conversely, a model
can ignore a fact that remains present. Separate missing project instructions,
truncated tool output, summary error, wrong environment, and provider context
overflow; increasing one budget does not repair every boundary.

## Sources

- [Instruction discovery](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/agents_md.rs),
  [configuration defaults](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/config/src/config_toml.rs).
- [Compaction routing](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/session/turn.rs),
  [local replacement history](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/compact.rs),
  [remote V2 compaction](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/core/src/compact_remote_v2.rs).

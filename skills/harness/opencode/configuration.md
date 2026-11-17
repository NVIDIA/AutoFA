# OpenCode — Configuration

**At a glance:** Capture effective merged settings, not only the project's JSON.
Values below describe the [inspected revision](overview.md); older releases can
have different defaults and tool names.

## How it works

Official configuration supports JSON/JSONC. Precedence runs from remote defaults
through global config, `OPENCODE_CONFIG`, project config, `.opencode` resources,
inline `OPENCODE_CONFIG_CONTENT`, then managed settings. Configurations merge rather
than replacing the entire object. The global file normally resides at
`~/.config/opencode/opencode.json`; `OPENCODE_CONFIG_DIR` changes resource discovery.

| Setting | Inspected behavior | Analysis use |
|---|---|---|
| `model`, `agent.<name>.model`, variants/provider options | Effective model route and generation parameters | Compare actual requests, including agent overrides |
| `agent.<name>.steps` | Unbounded when unset; injects a summary instruction at threshold | Not proof of a hard execution cutoff |
| `permission`, agent permissions | `allow` / `ask` / `deny`; per-tool/pattern policy | Establish availability and execution gate |
| `subagent_depth` | `1` default in task execution | Nested delegation rejection |
| `compaction.auto` | Enabled unless false | Context-overflow recovery |
| `compaction.prune` | Disabled unless enabled | Clearing old tool outputs |
| `compaction.preserve_recent_tokens`, `tail_turns`, `reserved` | Retention and input budget controls | Compare summary/retained context |
| `tool_output.max_lines`, `max_bytes` | `2000`, `51200` defaults | Truncated observations and saved output |
| Provider `options.headerTimeout`, `chunkTimeout`, `timeout` | Header/idle defaults 300,000 ms; full-request timeout separately configured | Distinguish network gaps from total duration |
| `instructions`, `plugin`, `mcp` | Extra prompt and execution inputs | Recover customized behavior |
| `experimental.continue_loop_on_deny` | Not enabled by default | Whether denial stops continuation |

## Where errors can originate

Collect the invocation, workspace, effective agent/provider configuration, plugin
versions, and enclosing runner limits. Record non-secret settings; credentials are
not needed for attribution. A project file alone cannot establish which managed,
inline, agent, or plugin override took effect. Treat experimental options as
revision-specific, including native LLM and code-mode selection.

## Sources

- [Official config and precedence](https://opencode.ai/docs/config/).
- [Core schema](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/core/src/v1/config/config.ts),
  [provider schema](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/core/src/v1/config/provider.ts),
  [agent defaults](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/agent/agent.ts).

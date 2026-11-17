# Codex — Configuration

**At a glance:** Capture resolved thread/turn settings as well as configuration
files. CLI overrides, profiles, project trust, managed requirements, and client
requests can change the effective configuration.

## How it works

User configuration normally resides at `$CODEX_HOME/config.toml` (`~/.codex` by
default); trusted project configuration, profiles, `-c key=value`, and managed
policy contribute additional layers. Managed restrictions can constrain otherwise
valid user overrides. Use effective settings from the run rather than treating
one config file as authoritative.

| Setting/interface | Meaning | Analysis use |
|---|---|---|
| `model`, `model_provider`, `model_providers.*` | Model route, API endpoint and provider options | Separate harness identity from model identity |
| `model_reasoning_effort`, output controls | Request-specific generation behavior | Truncated/slow responses and provider compatibility |
| `approval_policy`, sandbox/permission profile | Approval routing and executable filesystem/network scope | Explain denied or differently isolated calls |
| `model_context_window`, `model_auto_compact_token_limit` and scope | Input/window accounting | Establish actual compaction trigger |
| `model_post_turn_compact_threshold_percent` | Optional post-turn compaction | Explain work after apparent answer completion |
| `project_doc_max_bytes` | 32 KiB default | Project instruction truncation |
| `project_root_markers`, `project_doc_fallback_filenames` | Instruction search boundary | Missing/different project instructions |
| `tool_output_token_limit` | Model-facing tool output control | Distinct from command runtime and context window |
| Provider request/stream retry settings | Defaults 4 and 5; idle timeout 300,000 ms | Separate transport attempts and inactivity |
| `mcp_servers`, feature flags, hooks | Tool inventory and lifecycle customization | Missing tools, altered continuation, approvals |
| `--json`, `--output-schema`, `--output-last-message` | Event format, final-answer schema, final-text file | Three independent output choices |
| `--ephemeral` | No ordinary persistent session recording | Missing artifacts may be intentional |

The official exec guide documents a read-only default and explicit sandbox flags.
The inspected exec setup requests noninteractive approval behavior, subject to
managed configuration handling. Do not assume an interactive client's permission
profile or approval flow applies to a headless evaluation.

## Where errors can originate

Record invocation, client capabilities, effective config/profile, environment/cwd,
feature state, model metadata, and outer limits. A displayed model name or sandbox
label is insufficient to establish the full request or tool policy. Do not copy
credentials into analysis artifacts; non-secret routing settings and error records
are the relevant evidence.

## Sources

- [Official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference),
  [noninteractive behavior](https://learn.chatgpt.com/docs/non-interactive-mode).
- [Config schema/defaults](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/config/src/config_toml.rs),
  [exec configuration](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/exec/src/lib.rs),
  [provider settings](https://github.com/openai/codex/blob/0b1b78a4f1694e2b9e393d385c7b82ca714ca08a/codex-rs/model-provider-info/src/lib.rs).

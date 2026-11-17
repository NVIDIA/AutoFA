# OpenCode — Tool calling

**At a glance:** Normal tools use provider-native schemas and correlated tool-call
IDs. Tool availability, permission checks, schema validation, execution status, and
the command's exit status are different evidence surfaces.

## How it works

- The registry combines built-ins and plugin tools; session resolution adds MCP
  integration and applies the selected model/agent contract. Experimental code mode
  changes the exposed interface. Inspect the submitted schemas instead of assuming
  all installed tools were visible.
- `Tool.wrap` validates arguments with the tool's schema before execution and reports
  `ToolInvalidArgumentsError` with correction feedback. Plugin definitions retain
  their public Zod/JSON Schema boundary.
- Permission checks use merged agent/session rules. `allow`, `ask`, and `deny` have
  different effects; a waiting approval is not an executed command. Plugins can
  modify arguments and results through tool hooks.
- Tool parts progress through `pending`, `running`, `completed`, or `error`, with
  `callID`, input, timing, and terminal feedback. Cleanup marks unresolved tool calls
  as interrupted errors.
- The shell implementation returns `metadata.exit`, output, and timeout/abort text.
  A nonzero command exit can be carried by a normally returned tool result. Its
  default deadline is 120,000 ms, overridable per call and by the runtime flag.
- Generic output truncation defaults to 2,000 lines or 50 KiB. It saves the full
  output and returns a preview plus `metadata.truncated` and `outputPath`; tools
  providing their own truncation metadata bypass that generic pass.

## Where errors can originate

Preserve the tool name actually exposed by the evaluated revision: the inspected
source uses a shell implementation, while older traces/docs may say `bash`.
Separate missing tools, invalid arguments, denied permissions, nonzero exits,
timeouts, and absent dependencies. Inspect saved output before concluding the
command omitted required detail; a successful tool lifecycle is not task success.

## Sources

- [Registry](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/tool/registry.ts),
  [session tool resolution](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/tools.ts),
  [validation wrapper](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/tool/tool.ts).
- [Shell execution](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/tool/shell.ts),
  [truncation](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/tool/truncate.ts),
  [official permissions](https://opencode.ai/docs/permissions/).

# Pi — Configuration

**At a glance:** Pi merges agent-directory and trusted project settings; invocation
and SDK choices can override them. Capture the effective configuration for the
[inspected coding-agent path](overview.md).

## How it works

The agent directory defaults to `~/.pi/agent`, overridable with
`PI_CODING_AGENT_DIR` or SDK `agentDir`. It holds `settings.json`, `models.json`,
authentication, MCP configuration, instructions, and extensions. Project settings
live in `.pi/settings.json`; resources combine according to their own rules.
Project executable resources/settings load after trust, with `sessionDir` an
explicit early-read exception. Context files are discovered independently of trust.

| Setting | Inspected default/meaning | Analysis use |
|---|---|---|
| `defaultProvider`, `defaultModel`, model registry | Startup route; runtime selection can change | Record provider/API/model per response |
| `defaultThinkingLevel` | `medium` | Effective model-specific thinking/output budget |
| `defaultTools` | `read`, `bash`, `edit`, `write` | Plain names replace; `+`/`-` entries modify selection |
| `steeringMode`, `followUpMode` | `one-at-a-time` | Queue delivery, not tool parallelism |
| `compaction.enabled` | `true` | Automatic context reduction |
| `compaction.reserveTokens`, `keepRecentTokens` | `16384`, `20000` | Input headroom and retained tail; model overrides supported |
| `retry.enabled`, `maxRetries`, `baseDelayMs` | `true`, `3`, `2000` | Agent retry policy |
| `retry.maxAgentDelayMs` | `60000` | Agent backoff ceiling |
| `retry.provider.*` | Separate timeout/count/delay controls | Lower-level transport attempts |
| `httpIdleTimeoutMs` | `300000`; zero disables | Header/body inactivity |
| `sessionDir`, `--session-dir`, `PI_CODING_AGENT_SESSION_DIR` | Session location overrides | Locate persisted evidence |
| `--no-session` | Ephemeral run | No resumable JSONL after exit |
| `shellPath`, `shellCommandPrefix` | Optional execution customization | Shell environment and command behavior |

MCP uses agent/project `mcp.json`. Project entries replace same-named global servers;
exposure controls direct versus discovered/code-mode tools. Neither that file nor
the basic settings list fully describes installed extension behavior.

## Where errors can originate

Record CLI flags, trust decision, SDK overrides, extension versions, active tools,
model metadata, and enclosing time/token limits. An empty `defaultTools` disables
built-ins without necessarily disabling extension tools. Do not infer a global
tool deadline from HTTP retry settings; bash's optional timeout is per-call seconds.

## Sources

- [Configuration and trust](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/docs/configuration.md),
  [settings reference](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/docs/settings.md),
  [settings implementation](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/settings-manager.ts).

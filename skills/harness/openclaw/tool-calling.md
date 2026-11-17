# Openclaw — Tool calling

**At a glance:** Tools use **native provider function-calling** (schemas passed as `params.tools`). **Tool policy is enforced *before* the model call** — a removed tool's schema never reaches the model that turn. Two alternative modes exist: **code mode** (model writes JS run in a QuickJS-WASI VM) and experimental **tool search** (a single `tool_search_code` tool). Results flow back via the harness result shape + `observeToolTerminal`.

## How it works

**Primary format — native function-calling / tool-use:** runtimes "handle native tool calls"; tool schemas are passed into the prepared attempt as `params.tools`.

**Alternative calling modes:**
- **Code mode:** hides the catalog and exposes `exec` + `wait` + required direct-only tools; the model writes JavaScript evaluated in a **QuickJS-WASI** VM. **MCP tools are callable *only* via code mode** (`API.list("mcp")`, `MCP.<server>.<tool>()`).
- **Tool Search (experimental):** the model gets a single `tool_search_code` tool; it runs a short JS body in an **isolated Node subprocess** — used when the catalog is large.
- **Skills:** injected as an XML `<available_skills>` list; the model loads a skill with `read`.

**Core tools:** runtime execution (`exec`, `process`, `terminal`, `code_execution`), file I/O (`read`, `write`, `edit`, `apply_patch`), web (`web_search`, `web_fetch`, `x_search`, `browser`), messaging, sessions/subagents, automation (`cron`, `heartbeat_respond`), media, and catalog discovery (`tool_search`, `tool_describe`, `tool_search_code`).

**Policy enforcement (critical for eval):** "**Tool policy is enforced before the model call. If policy removes a tool, the model does not receive that tool's schema for the turn.**" Restrictions come from global/per-agent config, channel policy, provider restrictions, sandbox rules, or plugin availability. Precedence: `tools.deny` wins over `tools.allow`.

**Result feedback:** harnesses return results through the harness result shape and call `observeToolTerminal` on terminal state (reporting call id, tool name, arguments, whether execution started, and outcome). Hooks: `tool_result_persist` transforms results before they're written to the session transcript; `before_tool_call` / `after_tool_call` can intercept params/results.

## Where errors can originate

- **Policy-stripped tools** → the model never sees a tool's schema and can (a) "fail" a task it could otherwise do, or (b) **hallucinate the missing tool** and produce an unresolvable call. High-frequency, easy to misread as a model error.
- **Loop detection** (`tools.loopDetection`: `historySize`, `warningThreshold`, `criticalThreshold`) can abort/flag repetitive tool behavior — a run cut short mid-task.
- **Code-mode VM failures:** QuickJS-WASI execution errors surface as tool failures; MCP being reachable *only* via code mode means MCP tasks fail entirely if code mode is off.
- **`apply_patch` gating** (`tools.exec.applyPatch.{enabled,allowModels}`): patch tool silently unavailable to non-allowed models.

## Sources

- `https://docs.openclaw.ai/tools`, `https://docs.openclaw.ai/gateway/config-tools` — core tool catalog, policy enforcement, loop detection.
- `https://docs.openclaw.ai/reference/code-mode`, `https://docs.openclaw.ai/plugins/sdk-agent-harness` — code mode, QuickJS-WASI, harness result shape.

> Gap: the exact wire serialization of a tool *result* back into the next model message is not documented.

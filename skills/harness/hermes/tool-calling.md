# Hermes — Tool calling

**At a glance:** Tools use **native OpenAI function-calling** (JSON), translated per-provider before the call (Gemini/Moonshot schema sanitizers). Arguments are `json.loads`-parsed at execution; malformed args yield a synthetic error **without executing**. Three execution strategies (concurrent/sequential/segmented) exist, with an 8-worker pool, a 420 s batch timeout, and a guardrail controller that can `warn/block/halt` repeated failing calls.

## How it works

**Format.** Native OpenAI function-calling (JSON), **not** XML tags. The model returns `assistant_message.tool_calls`, each with `.function.name`, `.function.arguments` (a JSON **string**), and `.id`. Tool schemas follow the OpenAI shape `{"name","description","parameters": <JSON Schema>}`. Per the narrow-waist invariant, every model tool ships on every call (no per-turn stripping).

**Per-provider schema translation** (before the call):
- **Gemini:** strips to Gemini's allowed key subset and fixes typed-enum/`required` mismatches (`agent/gemini_schema.py`).
- **Moonshot/Kimi:** fills missing `type`, moves `type` onto `anyOf` children, strips `nullable` (`agent/moonshot_schema.py`).

**Argument parsing.** Arguments are `json.loads`-parsed and must be a dict; on failure a synthetic error result is returned *without executing*.

**Execution strategies.** Concurrent (thread pool), sequential, and segmented — all in `agent/tool_executor.py`.

**Result feedback.** Each result becomes a `role:"tool"` message appended in original order; incremental SessionDB flush after each. Writes/patches are tracked as mutating; other tools as no-effect or idempotent (for guardrail tracking).

**Loop guardrails.** Repeated failed / no-progress calls are tracked and return `allow|warn|block|halt`. Thresholds: `exact_failure_warn_after=2`, `exact_failure_block_after=5`, `same_tool_failure_warn_after=3`, `same_tool_failure_halt_after=8`, `no_progress_warn_after=2`, `no_progress_block_after=5`; `warnings_enabled=True`, `hard_stop_enabled=False` (**warnings only unless opted in**).

**Key defaults.** `_MAX_TOOL_WORKERS = 8`; concurrent-batch timeout **420 s** (env `HERMES_CONCURRENT_TOOL_TIMEOUT_S`, `<=0` disables). A `tool_search` bridge tool is "unwrapped" to the underlying tool with session-scope enforcement.

## Where errors can originate

- **Malformed tool arguments** → non-JSON or non-object args are **not executed**; a synthetic `{"error":"Invalid tool arguments",...}` is returned. (A separate *outbound* repair path also exists — see [parsing-and-output.md](parsing-and-output.md).)
- **Tool timeout** → a tool exceeding 420 s yields a synthetic result `status="timeout", error_type="tool_timeout"`.
- **Provider schema rejection** → one bad MCP tool schema can 400 the *entire* request pre-generation; the Gemini/Moonshot sanitizers exist specifically to prevent this.
- **Guardrail halt** → repeated identical failing calls can hard-stop the turn if `hard_stop_enabled` is set.
- **Invalid / empty tool name** → returns a terse error for blank names and the full catalog for a real typo.
- **Interrupt during batch** → pending tool calls are cancelled with `[Tool execution cancelled ...]` results.

## Sources

- `nousresearch/hermes-agent` `agent/tool_executor.py` — argument parsing, execution strategies, timeout, result messages.
- `nousresearch/hermes-agent` `agent/tool_guardrails.py`, `agent/gemini_schema.py`, `agent/moonshot_schema.py` — guardrail controller, per-provider schema sanitizers.

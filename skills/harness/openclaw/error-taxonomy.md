# Openclaw — Error taxonomy (harness-introduced failure modes)

**At a glance:** Keystone file for trace attribution. Each row maps a **symptom you'd see in a trace** → the **harness/Gateway mechanism** that caused it → the **config knob** that controls it → the **source**. Openclaw resolves a lot *before* the per-turn executor runs, so several "harness" failures are really Gateway-layer decisions (tool policy, compaction, failover).

## Failure-mode table

| # | Symptom in trace | Likely harness/Gateway cause | Knob / control | Source |
|---|---|---|---|---|
| 1 | Model ignores or hallucinates a tool it "should" have | **Tool policy stripped the tool's schema before the model call** | `tools.allow`/`tools.deny` (deny wins), `tools.byProvider`, `tools.toolsBySender`, channel/sandbox/provider policy | `/tools`, `/gateway/config-tools` |
| 2 | Context truncated / model "forgets" earlier turns | **Auto-compaction** summarized older turns (near limit or on context-overflow error → compact+retry) | `agents.defaults.compaction.keepRecentTokens=20000`, `mode="safeguard"`, `maxActiveTranscriptBytes`, `compaction.model` | `/concepts/compaction` |
| 3 | Instructions in `AGENTS.md`/`USER.md` not honored | **Bootstrap file truncation** (silent context loss) | `bootstrapMaxChars=20000`, `bootstrapTotalMaxChars=60000`, `bootstrapPromptTruncationWarning` | `/concepts/system-prompt` |
| 4 | Duplicated or missing turns after a retry | **Retry-induced dedupe:** buffers reset on retry; successor transcripts drop exact-duplicate long user turns in a short window | (compaction-driven retry path) | `/concepts/agent-loop`, `/concepts/compaction` |
| 5 | Run ends abruptly with provider error, no retry | **Non-rate-limit failures fail immediately, no key rotation** (only 429/quota/throttle rotate keys) | `agents.defaults.model.primary` + fallbacks | `/concepts/model-failover` |
| 6 | Run hangs then dies at a time boundary | **Timeouts:** run timeout enforced by `runEmbeddedAgent`; per-tool exec/web timeouts | `tools.exec.timeoutSec=1800`, web fetch/search `timeoutSeconds=30`, run timeout | `/concepts/agent-loop`, `/gateway/config-tools` |
| 7 | Long/looping task cut off mid-way | **Loop detection** aborts/flags repetitive tool behavior | `tools.loopDetection.{historySize,warningThreshold,criticalThreshold}` | `/gateway/config-tools` |
| 8 | Task never converges, ends only at timeout | **No hard max-iteration cap** — only stop conditions + timeout terminate the loop | (none documented) | `/concepts/agent-loop` |
| 9 | Wrong runtime used; no graceful fallback | **Fail-hard runtime selection** — a claimed run is not replayed; explicit `agentRuntime.id` fails closed (only `auto` falls back to `openclaw` compat runtime) | `agents.defaults.models[...].agentRuntime.id`, `OPENCLAW_AGENT_RUNTIME` (ignored under model/provider policy) | `/concepts/agent-runtimes`, `/plugins/sdk-agent-harness` |
| 10 | Observer/UI trace missing intermediate state | **Events are not replayed;** a dropped WebSocket event forces a client refresh | (short-lived dedupe cache) | `/concepts/architecture` |
| 11 | "Clean" terminal event hides an underlying failure | **Fallback tool-error reply / fallback lifecycle end/error** synthesized when the loop produced nothing renderable | — | `/concepts/agent-loop` |
| 12 | MCP task fails entirely | **MCP tools reachable only via code mode;** if code mode is off, they're unavailable | code mode enable; `tools.*` | `/reference/code-mode` |
| 13 | `apply_patch` unexpectedly unavailable | **Patch tool gated by model allowlist** | `tools.exec.applyPatch.{enabled,allowModels}` | `/gateway/config-tools` |
| 14 | Reply continues after "memory exhausted" | **Memory flush exhausted but reply continues** (documented compaction edge) | `compaction.*` | `/concepts/compaction` |

## Sources

- `https://docs.openclaw.ai/tools`, `https://docs.openclaw.ai/gateway/config-tools` — tool policy, loop detection, exec/patch, timeouts.
- `https://docs.openclaw.ai/concepts/compaction`, `https://docs.openclaw.ai/concepts/system-prompt` — compaction, retry, bootstrap truncation.
- `https://docs.openclaw.ai/concepts/model-failover`, `https://docs.openclaw.ai/concepts/agent-loop` — rate-limit-only retry, timeouts, no max-iteration cap.
- `https://docs.openclaw.ai/concepts/agent-runtimes`, `https://docs.openclaw.ai/reference/code-mode` — fail-hard runtime selection, MCP via code mode only.

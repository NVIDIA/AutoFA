# Openclaw — Configuration

**At a glance:** Config lives in **`~/.openclaw/openclaw.json`** (JSON5). The namespaces that most shape run outcomes are `agents.defaults.*` (models, compaction, bootstrap, context limits), `tools.*` (policy, exec, loop detection, sandbox), and `models.providers.*`. Several debug env vars expose the model transport / payload / SSE / code-mode internals.

## How it works

**Config file:** `~/.openclaw/openclaw.json` (JSON5). Key namespaces:

- **Runtime selection:** `agents.defaults.models["provider/model"].agentRuntime.id`, `models.providers.<provider>.agentRuntime.id`. Env `OPENCLAW_AGENT_RUNTIME` and session pins exist but are **ignored** under model/provider-scoped policy.
- **Models:** `agents.defaults.model.primary` + fallbacks; per-model `params.transport`, `params.extra_body`, `contextWindow`, `maxTokens`, `compat.*`, `reasoning`.
- **Tools:** `tools.profile` (`minimal|coding|messaging|full`), `tools.allow` / `tools.deny` (**deny wins**), `tools.byProvider`, `tools.toolsBySender`, `tools.exec.timeoutSec=1800`, `tools.exec.applyPatch.{enabled,allowModels}`, `tools.loopDetection.*`. Protected paths (`tools.exec.ask` / `tools.exec.security`) cannot be self-rewritten by the agent. See [error-taxonomy.md](error-taxonomy.md) for failure implications of each knob.
- **Prompt / agents:** `agents.defaults.workspace`, `skipBootstrap`, `bootstrapMaxChars=20000`, `bootstrapTotalMaxChars=60000`, `subagents.delegationMode` (`suggest`/`prefer`), `contextLimits.*`, `skillsLimits.maxSkillsPromptChars`, `userTimezone`, `timeFormat`.
- **Compaction:** `agents.defaults.compaction.{keepRecentTokens=20000, mode="safeguard", maxActiveTranscriptBytes, model}`.
- **Gateway:** bind default `127.0.0.1:18789`; start `openclaw gateway`; supervise via launchd/systemd.

**Debug env vars:** `OPENCLAW_DEBUG_MODEL_TRANSPORT=1`, `OPENCLAW_DEBUG_MODEL_PAYLOAD` (`summary|tools|full-redacted`), `OPENCLAW_DEBUG_SSE` (`events|peek`), `OPENCLAW_DEBUG_CODE_MODE=1`.

## Where errors can originate

- **Silent policy interactions:** `tools.deny` overriding `tools.allow`, `tools.byProvider`, and `tools.toolsBySender` can remove a tool for a specific model/sender only — reproducible failures that don't show up in other configs.
- **Ignored overrides:** `OPENCLAW_AGENT_RUNTIME` / session pins are **silently ignored** when a model/provider-scoped `agentRuntime.id` policy is set — a "why isn't my runtime override taking effect" trap.
- **Default caps bite silently:** `bootstrapMaxChars`/`bootstrapTotalMaxChars`, `keepRecentTokens`, `exec.timeoutSec` all have defaults that truncate/compact/kill without an explicit error unless you know to look.
- **Profile mismatch:** `tools.profile=minimal` on a coding task removes the tools the model needs.

## Sources

- `https://docs.openclaw.ai/gateway/config-tools` — `tools.*`, exec, loop detection, applyPatch, elevated, sandbox.
- `https://docs.openclaw.ai/concepts/system-prompt`, `https://docs.openclaw.ai/concepts/compaction` — bootstrap caps, context limits, compaction knobs.

# OpenCode — Model interaction

**At a glance:** Provider selection and request transforms precede streaming.
OpenCode's processor retry policy is distinct from SDK retries, context recovery,
model selection, and evaluation deadlines.

## How it works

`LLMRequestPrep.prepare` merges model/agent options and variants, applies plugin
hooks, and prepares tools, headers, system text, and messages. `LLM` routes through
the AI SDK adapter by default; the experimental native path logs its selected
runtime or fallback reason. The GitLab workflow integration has additional execution
and approval handling. Identify the concrete route before extrapolating behavior.

The AI SDK `streamText` call sets `maxRetries` to `input.retries ?? 0`. The enclosing
`SessionProcessor` uses `SessionRetry.policy`, allowing five retries for classified
transient errors at this revision. Classification considers API retryability,
server status, and selected rate-limit/network/overload messages. Context overflow
is excluded and routed to compaction instead.

Backoff starts at 2 seconds, doubles with up to 25% added jitter, and honors
`retry-after-ms` or `retry-after`. The 30-second fallback ceiling applies without
response headers; it is not an unconditional ceiling on every retry wait.
Retry status carries `attempt`, `message`, and `next`.

Provider options distinguish whole-request `timeout`, response `headerTimeout`,
and streamed-chunk `chunkTimeout`. Header and idle-chunk defaults are 300,000 ms
in the inspected transport; `false` disables the respective timeout. These do not
set a shell deadline or guarantee that an external wrapper will wait as long.

## Where errors can originate

Record effective provider/model, endpoint, adapter, non-secret options, error body,
and retry timing. A recovered 429 is not itself the terminal failure. Do not infer
automatic credential rotation or cross-model failover from the request retry loop.
Partial stream output may precede an error and another request; use IDs and statuses
to distinguish it from independent model decisions.

## Sources

- [LLM dispatch](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/llm.ts),
  [request preparation](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/llm/request.ts).
- [Retry classification and schedule](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/session/retry.ts),
  [provider timeouts](https://github.com/anomalyco/opencode/blob/7945de208964a49300d7f770d1a71d078db9a4c4/packages/opencode/src/provider/provider.ts).

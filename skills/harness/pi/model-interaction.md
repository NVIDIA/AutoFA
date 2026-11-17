# Pi — Model interaction

**At a glance:** The coding agent calls `ModelRuntime.streamSimple` through the
provider library. Agent-level retry, provider transport retry, model routing, and
compaction recovery are independently configurable.

## How it works

The SDK builds request options from effective settings and routes the selected
model, API, transport, reasoning controls, and credentials through `ModelRuntime`.
Extensions can transform the provider payload and observe response headers and
provider-stream events. Virtual-model or custom-provider configuration may change
the route; preserve the resolved model on each assistant message.

Automatic agent retry is enabled by default, with `maxRetries=3`,
`baseDelayMs=2000`, and a 60,000 ms agent-delay ceiling. Classification distinguishes
transient provider/network failures from quota/billing exhaustion and aborts.
Context overflow is routed to compaction instead of ordinary retry. The failed
assistant attempt is retained in raw history and omitted from model projection
before continuation. A successful response resets the retry counter.

Compaction and branch-summary calls use the same retry settings with separate
summarization lifecycle events. Provider retry settings (`retry.provider.*`) govern
lower-level attempts; documented provider retry count defaults to zero. Do not
multiply or combine these counters into an iteration cap without tracing the path.

`httpIdleTimeoutMs` defaults to 300,000 ms; provider request and WebSocket connection
timeouts are separate controls. A quiet stream, a whole request, a shell process,
and the external benchmark each have different deadline semantics.

## Where errors can originate

Match `auto_retry_start`/`auto_retry_end` and summarization events to the error that
triggered them. A temporary error followed by successful recovery is not proof of
task failure. Model changes must be supported by routing/session evidence; retries
alone do not imply automatic model failover. Token/cost totals may include cache
warming and summaries that are not task-solving assistant turns.

## Sources

- [Request options and provider hooks](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/sdk.ts),
  [session retry](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/agent-session.ts).
- [Retry classification and delay](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/ai/src/utils/retry.ts),
  [settings implementation](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/src/core/settings-manager.ts),
  [network/retry settings](https://github.com/earendil-works/pi/blob/33e203354391ba26937cd1e8a210b4db10e82c66/packages/coding-agent/docs/settings.md).

# LangChain — Model interaction

**At a glance:** Model transport belongs to the selected integration. Retries can
occur in the provider SDK, model wrapper, agent middleware, graph task, or outer
runner; count those layers separately.

## How it works

The factory accepts a model instance or resolves a model string with
`init_chat_model`. It binds the effective tools and structured-output strategy,
then invokes the model through installed middleware. Middleware can change the
model or request dynamically. Provider integrations convert responses into
LangChain messages and metadata; finish reasons, token usage, timeouts, and hosted
tool results are integration-specific.

| Retry/fallback layer | Inspected contract | Evidence to collect |
|---|---|---|
| Provider integration/SDK | No universal LangChain timeout or retry default | Integration version, constructor/client settings, HTTP attempts |
| `ModelRetryMiddleware` | Optional; default two retries after the first attempt, 1 s initial delay, factor 2, 60 s cap, jitter | Installed instance, matching exceptions, attempts, `on_failure` |
| `ToolRetryMiddleware` | Optional; same default retry count and delay settings | Per-tool attempts and side effects |
| `ModelFallbackMiddleware` | Optional ordered alternatives | Actual model for each request, exception and wrapper order |
| LangGraph `RetryPolicy` | When configured: three total attempts by default, 0.5 s initial interval, factor 2, 128 s cap, jitter | Task/graph policy and retry predicate |

The factory's middleware list is empty by default. A `RetryPolicy` type existing
in the library does not install it on every task: the graph retry path re-raises
when there is no policy. Graph retries clear pending task writes, not external
effects already performed by a tool.

Retry middleware defaults to `on_failure="continue"` for exhausted matching errors;
this can return fallback feedback instead of propagating the original exception.
Non-matching exceptions still propagate. Inspect the returned message and installed
policy before deciding whether a normally completed graph recovered successfully.

## Where errors can originate

Distinguish a provider attempt from a graph step and a new model decision. Nested
retry policies can multiply requests and latency; fallback can change tool or
structured-output capabilities. A context-window error requires inspection of the
actual request and integration limits. Do not import another harness's automatic
compaction or failover policy into a plain LangChain agent.

## Sources

- [Model request path](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/factory.py),
  [model retry](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/middleware/model_retry.py),
  [tool retry](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/middleware/tool_retry.py),
  [model fallback](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/middleware/model_fallback.py).
- [Graph retry execution](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/langgraph/langgraph/pregel/_retry.py),
  [RetryPolicy defaults](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/langgraph/langgraph/types.py).

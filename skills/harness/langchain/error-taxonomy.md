# LangChain — Error taxonomy

**At a glance:** These are investigation categories, not a LangChain error enum or
ground-truth failure labels. Prove the mechanism against the evaluated application's
graph, dependency versions, and configuration.

## How to use this file

Locate the first incorrect boundary, preserve its exact input/result, and inspect
later recovery. Distinguish task failure from a handled local error, an intentional
interrupt, or a truncated export. Read the same-named topic files for other harnesses
only after establishing the LangChain variant actually used.

| Candidate category | Observable symptom | Confirm before attribution | Relevant control/source |
|---|---|---|---|
| Tool visibility/dispatch mismatch | Unknown tool or no expected call | Effective bound tools, middleware filtering, provider versus local execution | Factory; [tool calling](tool-calling.md) |
| Tool invocation failure | Error ToolMessage before function execution | Call ID, arguments, schema, injected dependencies, validation exception | ToolNode; `ToolInvocationError` |
| Tool execution failure | Raised exception, failed span, or error feedback | Tool implementation, custom handlers, retries, external side effects | `BaseTool`; tool middleware |
| Structured-response failure | Repeated response-tool calls, validation error, missing result | Actual strategy, schema type, `handle_errors`, terminal state | [Parsing and output](parsing-and-output.md) |
| Provider/transport failure | HTTP error, timeout, backoff, fallback | Integration and SDK versions, request attempts, selected model | [Model interaction](model-interaction.md) |
| Graph/budget termination | `GraphRecursionError`, limit feedback, external cancellation | Supersteps versus call counts; invocation limits and middleware budgets | Factory config; call-limit middleware |
| Context/state loss | Earlier evidence absent from later reasoning | Rendered request, same-ID replacement, deletion, summary, artifact visibility | Reducers; [prompt and context](prompt-and-context.md) |
| Pending interrupt mishandled | Run pauses or wrapper reports completion too early | Interrupt payload, checkpoint, thread ID, resume command | LangGraph interrupt/HITL middleware |
| Duplicate side effect | Repeated write or remote action | Task retry/resumption, pre-interrupt code, idempotency, attempt IDs | Graph retry and restart semantics |
| Persistence/thread mismatch | Wrong history, missing continuation, stale state | `thread_id`, checkpoint namespace/ID, backend and invocation | Checkpointer; [trace format](trace-format.md) |
| Export interpretation gap | Missing tools, wrong order, apparent duplicate tokens | Stream API/version/mode, subgraph namespace, parent/child runs, chunk assembly | Streaming and tracing adapters |
| Unsupported success claim | Normal graph completion but evaluator fails | Final artifacts and independent checks, not merely run status | Application result mapping and evaluator |

## Where errors can originate

Do not infer an installed policy from a class available in the package. Middleware
ordering, provider integration, custom graph nodes, and wrappers can change the
default contract. A successful retry can resolve a local fault; fallback feedback
can also allow normal graph termination without achieving the task.

Report the task/repeat, graph/node or run IDs, call IDs, exact supporting evidence,
mechanism and revision, plausible alternatives, and a focused validation step.
Keep attribution uncertain when the available trace cannot distinguish causes.

## Sources

- [Factory and routing](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/factory.py),
  [model call limits](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/middleware/model_call_limit.py).
- [ToolNode](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/prebuilt/langgraph/prebuilt/tool_node.py),
  [interrupt and retry contracts](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/langgraph/langgraph/types.py).

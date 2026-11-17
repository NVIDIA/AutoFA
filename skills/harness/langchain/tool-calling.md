# LangChain — Tool calling

**At a glance:** Client tool calls appear in `AIMessage.tool_calls`; results are
`ToolMessage` objects linked by `tool_call_id`. Provider-hosted tools follow the
provider's contract and need not traverse the local ToolNode.

## How it works

The factory accepts callables, `BaseTool` instances, and provider tool dictionaries.
Middleware can contribute tools or filter/change the request. ToolNode injects
runtime, state, or store dependencies separately from model-generated arguments.
Calls can run concurrently; tool declaration order does not imply serial execution.

| Boundary | Inspected behavior | Analysis implication |
|---|---|---|
| Argument schema | `BaseTool` supports Pydantic or JSON Schema; callable tools derive schemas | Inspect the registered tool and actual validation path |
| Unknown tool | ToolNode returns an error `ToolMessage` | Schema visibility and dispatched name may differ |
| Invocation validation | Default ToolNode handler converts `ToolInvocationError` to feedback | A call can fail before its function executes |
| Ordinary execution exception | Default ToolNode handler re-raises | Do not assume every exception becomes a model observation |
| Custom handling | `BaseTool` error settings, ToolNode handlers, and middleware can intercept | Determine which layer produced the terminal result |
| Result payload | `content` is model-facing; `artifact` can hold data not sent to the model | An exported artifact is not proof of model visibility |
| Result status | `ToolMessage.status` is `success` or `error`, defaulting to success | Status alone cannot establish semantic task success |
| State/routing output | Tools can return `Command` updates | Inspect state changes in addition to message text |

The factory constructs ToolNode without overriding its default error handler.
That default is revision-sensitive: invocation errors receive feedback, while
ordinary execution errors propagate. Graph interrupts propagate through a separate
control-flow path. Explicit tool-error/retry middleware can change the outcome.

## Where errors can originate

Record call ID, tool implementation, submitted and injected arguments, validation,
execution attempt, result status, and subsequent model request. Shell execution,
timeouts, output truncation, sandboxing, and permissions are application/tool
choices; generic `create_agent` does not establish their defaults. A retry can
repeat a write or remote action even if the graph discarded a failed task's writes.

## Sources

- [Factory tool setup](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/factory.py),
  [BaseTool contracts](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/core/langchain_core/tools/base.py),
  [ToolMessage fields](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/core/langchain_core/messages/tool.py).
- [ToolNode dispatch, injection, and default error handling](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/prebuilt/langgraph/prebuilt/tool_node.py).

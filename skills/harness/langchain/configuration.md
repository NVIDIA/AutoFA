# LangChain — Configuration

**At a glance:** Configuration normally lives in application code, dependency
versions, invocation config, and provider settings. There is no universal LangChain
agent config file whose contents fully determine a run.

## How it works

| Surface | Inspected modern-agent contract | Analysis use |
|---|---|---|
| `model`, `tools`, `system_prompt` | Required model; optional tools/prompt | Recover actual model instance, registered tools, and prompt |
| `middleware` | Empty sequence by default | Record installed instances, order, and overridden settings |
| `response_format` | None by default; schema or explicit strategy | Separate text completion from typed response handling |
| `state_schema`, `context_schema` | Optional application extensions | Inspect reducers, request data, and runtime context |
| `checkpointer`, `store` | None by default | Persistence and application memory are explicit choices |
| `interrupt_before`, `interrupt_after` | None by default | Static pause points differ from dynamic tool/HITL interrupts |
| Invocation `recursion_limit` | Factory supplies 9,999 unless overridden | Graph superstep budget, not a model-call count |
| Invocation `configurable.thread_id` | Identifies persisted thread when checkpointing | Wrong IDs can mix or lose intended conversation state |
| Model/tool call limits | Optional middleware with run/thread controls | Establish the installed limits and exhaustion behavior |
| Model/tool retry | Optional; two retries by default when installed | Separate retries from original calls and SDK attempts |
| Summarization | Optional; configured model, triggers, and retention | No universal compaction threshold in a plain agent |
| Tracing | LangSmith/environment/callback configuration | Establish whether intermediate runs were actually recorded |

Model integration parameters control credentials, endpoint, sampling, transport,
and timeouts. Record non-secret effective values and integration package versions.
`LANGSMITH_TRACING` and related LangSmith settings configure observability, not the
agent's task-success criteria. Tools and middleware may have additional independent
configuration and environment variables.

Legacy `AgentExecutor` defaults differ: `max_iterations=15`,
`max_execution_time=None`, `early_stopping_method="force"`,
`handle_parsing_errors=False`, and `return_intermediate_steps=False`. These names
must not be presented as `create_agent` constructor options.

## Where errors can originate

Constructor defaults alone are insufficient when `.with_config`, per-invocation
overrides, middleware, or wrappers intervene. Check installed package versions
against the [inspected snapshots](overview.md), especially LangGraph/prebuilt and
provider integration packages. Capture the graph and tool code alongside settings.

## Sources

- [Factory signature and compiled config](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/factory.py),
  [legacy executor configuration](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain/langchain_classic/agents/agent.py).
- [Official middleware guide](https://docs.langchain.com/oss/python/langchain/middleware/built-in),
  [LangSmith tracing setup](https://docs.langchain.com/langsmith/observability-quickstart),
  [checkpoint interface](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/checkpoint/langgraph/checkpoint/base/__init__.py).

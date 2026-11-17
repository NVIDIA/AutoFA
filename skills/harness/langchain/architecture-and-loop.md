# LangChain — Architecture and loop

**At a glance:** `create_agent` compiles a state graph, with model and tool nodes
plus middleware hooks. A graph step, a model request, and a tool attempt are
different units of work.

## How it works

1. Resolve the model, tools, response strategy, state schema, and middleware.
2. Build a model request from state and run the model through installed wrappers.
3. Merge the response into state. Pending client tool calls route through `Send`
   to tool tasks, with results correlated by tool-call ID.
4. Route back to the model for follow-up, or end when there is no pending work,
   a structured response is ready, or middleware explicitly redirects execution.

The tool-to-model edge can end for `return_direct` tools, but in this factory it
requires **all executed client-side tools in the batch** to be `return_direct`.
An assistant message with calls whose results already exist can instead cause
another model step. Tool-produced `Command` objects and middleware jumps can
alter state/routing; inspect those before reconstructing a simple alternating loop.

The factory sets `recursion_limit=9_999`. This bounds graph supersteps, not exactly
9,999 model calls or tools; invocation configuration can override it. Reaching the
limit raises `GraphRecursionError`. Optional model/tool call-limit middleware adds
separate run/thread budgets. External task deadlines remain separate.

With checkpointing, `interrupt()` pauses execution and exposes a value for the
caller. Resuming with `Command(resume=...)` restarts the interrupted node from its
beginning. Work before the interrupt can run again. Checkpoint recovery and task
retries do not guarantee external side effects happen exactly once.

## Where errors can originate

Inspect graph edges, state updates, middleware order, pending tasks, and invocation
config before attributing repetition to the model. Parallel tool tasks can complete
out of order. A wrapper that treats an interrupt as completion can lose required
approval/resumption. Legacy `AgentExecutor` uses its own loop with a default
15-iteration limit; that is not the modern factory's contract.

## Sources

- [Factory, routing, compilation, and recursion config](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/factory.py).
- [Graph execution](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/langgraph/langgraph/pregel/main.py),
  [interrupt and Command contracts](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/langgraph/langgraph/types.py),
  [official interrupt guide](https://docs.langchain.com/oss/python/langgraph/interrupts).
- [Legacy AgentExecutor](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain/langchain_classic/agents/agent.py).

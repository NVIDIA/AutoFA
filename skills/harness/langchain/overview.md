# LangChain — Overview

**At a glance:** LangChain's Python `create_agent` builds a LangGraph execution
graph around a chat model, tools, and optional middleware. The application's graph,
model integration, and middleware are part of the harness being evaluated.

## How it works

- This pack covers the current Python `langchain.agents.create_agent` factory and
  its LangGraph runtime. It distinguishes legacy `langchain_classic` agents where
  their contracts differ. JavaScript agents and custom graphs require their own
  implementation evidence.
- The graph alternates model and tool work, updates typed state, and follows edges
  until completion, interruption, or failure. Middleware can change prompts,
  tools, state, routing, retries, and termination.
- Provider integrations implement model transport and response conversion. There
  is no single LangChain provider API, shell environment, permission policy, or
  model context window.
- Checkpointers, stores, and LangSmith tracing are optional integrations. Deep Agents
  adds its own planning, filesystem, delegation, and context behavior; do not
  assume those features from a `create_agent` invocation alone.

## Instructions for the analysis agent

Capture package versions/lockfile, factory or graph definition, model integration,
tool implementations, middleware order, invocation config, and evaluator limits.
Read [trace-format.md](trace-format.md) and [error-taxonomy.md](error-taxonomy.md).
Identify the first incorrect boundary using actual inputs, outputs, and later
recovery. Final graph state or a successful LangSmith run is not evaluator success.
These references do not change AutoFA's scoring, evidence rules, or available tools.

## Where errors can originate

Separate model decisions, provider conversion, graph routing, reducers, middleware,
tool execution, persistence, and trace export. A resumable interrupt is not itself
a failure; a missing intermediate event is not proof that execution never occurred.
Do not apply legacy `AgentExecutor` limits or parser settings to `create_agent`.

## Sources

Inspected on 2026-09-29:

- LangChain `04ac76c07ec173a44e3e57de54861d0b637e3299`, whose
  [Python package](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/pyproject.toml)
  declares version `1.4.3` and LangGraph `>=1.2.11,<1.3.0`.
- LangGraph `07b33185eab893be2ed031eedae52f09314bf77c`, whose
  [runtime package](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/langgraph/pyproject.toml)
  declares version `1.2.12`. These are independently inspected source snapshots,
  not an integration-tested installation; the evaluated environment is authoritative.
- [Official agents guide](https://docs.langchain.com/oss/python/langchain/agents),
  [factory implementation](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/factory.py).

All implementation links in this pack pin these revisions. Check the run's exact
versions before applying defaults or event contracts.

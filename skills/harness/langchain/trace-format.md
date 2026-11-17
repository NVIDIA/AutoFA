# LangChain — Trace format

**At a glance:** Graph streams, runnable events, checkpoints, and LangSmith traces
are distinct representations. LangChain does not impose one universal agent JSONL
file; identify the producing API, version, and export adapter first.

## How it works

| Representation | Shape and meaning | Limits on interpretation |
|---|---|---|
| Graph `stream` / `astream` | `updates` emits node updates; `values` emits state; `messages` emits message chunks with metadata; `custom` carries application data | Selected mode determines visibility; subgraph/multiple-mode options affect envelopes |
| Runnable `astream_events(version="v2")` | Events include `event`, `name`, `run_id`, `parent_ids`, `tags`, `metadata`, and `data` | Nested starts/streams/ends are not one flat conversation; retain run hierarchy |
| New event streaming, `version="v3"` | Typed projections exposed by `stream_events` / `astream_events`, including message/tool views and final output | Different contract from v2 dictionaries; inspect consumer/export conversion |
| Graph checkpoint/state snapshot | State values, next tasks, config, metadata, parent information, tasks/interrupts; backend also tracks pending writes | Saved state is not a complete provider-wire or side-effect log |
| LangSmith trace | Optional traced root/child runs with inputs, outputs, errors, metadata, and timing | Export, redaction, sampling, or missing instrumentation can omit evidence |

Model messages can contain `tool_calls` and `invalid_tool_calls`; tool messages
carry `tool_call_id`, content, status, and optional artifact. Preserve message IDs
and chunk assembly: partial argument fragments are not final calls, and reducer
updates can replace messages with the same ID. Tool artifacts need not have been
sent to the model. Usage metadata and provider response metadata depend on the
integration and may be incomplete in a selected stream mode.

For persisted continuations retain `thread_id`, `checkpoint_ns`, `checkpoint_id`,
parent configuration, pending tasks, and interrupt values. Reconstruct the chosen
checkpoint lineage rather than concatenating every saved snapshot. Subgraph
namespaces and node metadata help distinguish concurrent work. Checkpoint formats
depend on the backend; do not assume a filesystem transcript exists.

## Where errors can originate

Recover the invocation input, effective model requests where recorded, terminal
tool results, final state, and evaluator artifacts. A final-state-only export can
omit retries, deleted messages, summaries' source context, and raw tool artifacts.
Repeated stream chunks need not be repeated model calls. A paused graph awaiting
resume is not a successful terminal answer, even when an outer request returns.

## Sources

- [Official streaming guide](https://docs.langchain.com/oss/python/langchain/streaming),
  [runnable streaming APIs](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/core/langchain_core/runnables/base.py),
  [v2 event implementation](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/core/langchain_core/tracers/event_stream.py).
- [Graph streaming](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/langgraph/langgraph/pregel/main.py),
  [state/interrupt contracts](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/langgraph/langgraph/types.py),
  [checkpoint storage contracts](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/checkpoint/langgraph/checkpoint/base/__init__.py).

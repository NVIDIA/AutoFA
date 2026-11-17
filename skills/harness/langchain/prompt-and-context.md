# LangChain — Prompt and context

**At a glance:** State, checkpoint history, and the messages actually sent to the
model are different views. Middleware can transform a request without preserving
that exact request in graph state.

## How it works

`create_agent` accepts an optional `system_prompt`, converted to a `SystemMessage`
when given as text. The model execution path prepends the effective system message
to request messages. Middleware can set dynamic prompts, edit message lists,
select tools/models, or update state before and after a request. The factory does
not itself discover repository `AGENTS.md` files or supply a universal coding prompt.

The agent's messages use LangGraph's `add_messages` reducer. Messages with new IDs
append; matching IDs replace existing messages. `RemoveMessage` deletes messages,
including the special remove-all operation. A state update is therefore not always
an append-only transcript. Custom state fields have their own reducer contracts.

Summarization and context editing are optional middleware, not automatic behavior
of every agent. Summarization evaluates configured triggers and retention rules,
creates a summary, and replaces history with summary messages plus retained recent
messages. Its cutoff logic preserves related assistant/tool messages. If enabled,
its default retention is 20 recent messages; the model, trigger, token counter,
and actual cutoff still matter. Summary generation is another model interaction
that can fail or omit details.

A checkpointer retains graph state across invocations associated with a thread.
A store is a separate facility for application-managed information. Neither proves
that all saved information was included in a particular provider request. Runtime
context and injected tool dependencies are also distinct from model-visible text.

## Where errors can originate

Compare the request before and after middleware and provider conversion. Look for
same-ID replacement, explicit deletion, summary boundaries, filtered tools, missing
artifacts, and application history selection. Attribute context loss only when the
needed information is absent from the effective request, not merely from a final
state export. Model context windows and output budgets come from the selected
integration/model and caller settings.

## Sources

- [Request assembly](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/factory.py),
  [summarization middleware](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/middleware/summarization.py),
  [context editing](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/middleware/context_editing.py).
- [Message reducer](https://github.com/langchain-ai/langgraph/blob/07b33185eab893be2ed031eedae52f09314bf77c/libs/langgraph/langgraph/graph/message.py),
  [official persistence guide](https://docs.langchain.com/oss/python/langgraph/persistence).

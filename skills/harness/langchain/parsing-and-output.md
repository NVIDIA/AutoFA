# LangChain — Parsing and output

**At a glance:** Provider message conversion, tool argument validation, and final
structured-response validation are separate boundaries. Successful JSON parsing
does not establish task correctness or even schema validation.

## How it works

Ordinary model output is represented by `AIMessage`, including content, tool calls,
and potentially invalid tool calls. At this revision the factory patches invalid
tool calls with error `ToolMessage` feedback when an unanswered call ID exists;
it skips entries without IDs. This supports another model attempt, not guaranteed
argument repair or successful execution.

When `response_format` is supplied, the factory uses one of these strategies:

| Strategy | Contract | Failure/recovery behavior |
|---|---|---|
| `AutoStrategy` / schema passed directly | Select provider-native output when supported, otherwise tool strategy | Capability metadata and actual selected model matter |
| `ProviderStrategy` | Request native structured output; parse final content when no tool calls remain | Invalid output raises `StructuredOutputValidationError` in this path |
| `ToolStrategy` | Bind synthetic response tools and extract their arguments | Multiple outputs or validation errors can create feedback and retry according to `handle_errors` |

`ToolStrategy.handle_errors` defaults to true. It can instead be false, a message,
an exception filter, or a callable. Pydantic, dataclass, and TypedDict schemas use
local typed validation. **Raw JSON Schema dictionaries are not locally validated
by ToolStrategy** at this revision: their argument dictionaries pass through. Do
not infer schema compliance merely because `structured_response` was populated.

The completed state can include `structured_response` alongside messages. That
field is distinct from ordinary tool output and from the evaluator's expected
answer. Streaming chunks may contain partial text or partial tool arguments;
inspect the assembled terminal message before judging malformed output.

## Where errors can originate

Check the provider response, message conversion, selected response strategy,
schema type, exact validation exception, and subsequent feedback. Legacy
`AgentExecutor.handle_parsing_errors` controls a different output-parser loop and
defaults to false; it does not configure `create_agent`. A fallback model or
middleware-selected model can change which structured-output path executes.

## Sources

- [Factory output handling and invalid-call patching](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/factory.py),
  [strategy definitions and schema validation](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain_v1/langchain/agents/structured_output.py).
- [Official structured-output guide](https://docs.langchain.com/oss/python/langchain/structured-output),
  [legacy parser handling](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/langchain/langchain_classic/agents/agent.py).

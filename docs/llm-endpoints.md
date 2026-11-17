# LLM endpoints

[Back to README](../README.md) · [Credentials and configuration](configuration.md) · [Detector reference](detectors.md)

Open **Settings > LLM endpoints** from the home menu, or run `autofa settings`.
Choose the role you want to configure:

| Submenu | Purpose | Default |
|---|---|---|
| **AutoFA harness** | Runs discovery and writes detectors; choose the harness, model, and thinking settings. | Authenticated CLI: Codex · GPT-6 Astra or Claude Code · Claude Opus 5.5 |
| **Judge endpoint** | Checks detector behavior and runs agentic taxonomy passes. Choose its harness, model, and thinking settings. | Same as AutoFA · GPT-6.1 Sol or Claude Sonnet 5.5 · Medium |
| **Detector LLM endpoint** | Runs the LLM calls inside detectors; choose the provider, URL, model, and key variable. | NVIDIA · GPT-6 Luna |

Model choices come from the selected installed harness: Claude Code reports its
model picker and capabilities; Codex reports its models and reasoning effort levels.
Switching harnesses switches the list and preserves your selections while browsing.
This reads metadata without running an inference request. **Custom model ID** lets
you enter a deployment ID, including when model discovery is unavailable. Existing
pinned IDs are preserved when the harness also exposes an alias for that model.

**Thinking** offers **Model default** or **On**. Adaptive Claude models expose their
supported **Effort** levels; models with fixed-budget thinking expose a **Token budget** when thinking
is on and an **Off (if supported)** choice. Adaptive models can require thinking,
so their controls do not promise an off switch or a fixed token budget. Codex has
its own model-specific effort choices. These controls affect actual discovery and
all behavior-judge stages, including claim compilation and confirmation.

The judge's **Harness** defaults to **Same as AutoFA**, with a separate model and
effort selection. Choosing Codex for discovery also switches an automatic judge
to a Codex model. You can explicitly select Claude Code or Codex in **Judge endpoint**
to use a different judge. Each uses its CLI's authentication and model catalog;
there is no separate HTTP URL for harness calls. Model access depends on your
account and harness / gateway configuration.

Taxonomy classification has two judges. Passes set to `llm` use the HTTP
taxonomy judge model configured by `judge_model`. Passes set to `agentic`, whether
the first pass or the second, use the Judge endpoint harness and model described
above. AutoFA does not silently fall back from an agentic pass to the HTTP judge
when the harness is unavailable or unauthenticated.

Codex support is validated with CLI **0.160.1**. Update the CLI if the
app-server rejects an option; restricted judging requires environment isolation.
Codex supports interactive and batch discovery, local slash commands,
AutoFA tool calls, interruption, and persisted thread resume through the
[Codex app-server protocol](https://learn.chatgpt.com/docs/app-server). Set
`codex_cli` in TOML (or `AUTOFA_CODEX_CLI`) for a custom executable; otherwise it
uses `codex` on PATH. Switching harnesses on an existing run starts a conversation
for the new harness while retaining its library and operation checkpoints.

Codex discovery automatically recovers from a model-capacity error (`serverOverloaded`).
After one same-model retry, it selects the next lower model in AutoFA's preference
order that the installed Codex catalog advertises: GPT-6 Astra, GPT-6.1 Sol, GPT-6 Sol,
GPT-5.6 Sol, then GPT-6 Luna. Unknown or specialist models are not guessed from names
or catalog order. At most two model switches occur during one discovery turn; ordinary
transient failures retain their two retries. Cancellation, authentication, quota,
and invalid-request errors do not trigger a model switch.

The CLI names each switch and displays the active model. The same conversation,
round and saved detector work continue, and Resume remembers the fallback for that
run. The requested model remains in settings; new runs start with it. Reasoning
effort is preserved when supported, otherwise the fallback uses its advertised
default and reports the change. Switches are recorded in the run's status log and
`codex_model.json`; mixed-model runs do not supply timing calibration data. This
policy applies to discovery only: judge and detector inference models stay configured.
Catalog inclusion does not guarantee account access or current capacity; if recovery
fails, saved work remains available with `autofa resume --run RUN_ID`.

Both harnesses use native conversation compaction. Use **`/compact`** between
interactive turns to make room before exploring more long traces; it preserves the
same conversation and saved detector work. If a discovery turn reaches its context
limit, AutoFA first asks that harness to compact and retries within the same session.
Native automatic compaction remains in charge of its model-specific thresholds.
Read large traces through the paginated search and evidence tools instead of
loading an entire trace into the conversation.

To select Codex directly in TOML:

```toml
harness = "codex"
model = "gpt-6-astra"
effort = "medium"  # Or a level advertised by this model; default delegates to Codex.
behavior_judge_harness = "auto"
behavior_judge_model = "gpt-6.1-sol"
behavior_judge_effort = "medium"
```

The model menu stays current with the installed harness; these IDs are configuration
examples. When switching harnesses in TOML, untouched shipped defaults resolve to
GPT-6 Astra for Codex discovery and GPT-6.1 Sol for its judge, or Opus/Sonnet for
Claude Code. Explicit custom deployment IDs are preserved. Existing configurations
keep their saved harness and model choices.

Under **Detector LLM endpoint**, choose:

- **Configured default** — uses the `hub_base_url` and `hub_api_key_env` that a
  deployment preset provides. AutoFA has no built-in default, so without a preset
  the form opens on **Other provider**.
- **Other provider** — enter the server's API root, model ID, and
  API key environment variable name.
- **Local server** — defaults to `http://localhost:8000/v1` with no API key.
  Start vLLM or SGLang separately and enter its served model ID.

Use **Tab** or **↑/↓** to move between fields and **←/→** to choose a provider.
The **Key variable** field takes an environment variable name, not the secret;
leave it blank for a server without authentication. **Save** / **Ctrl+S** writes
the form to the active `autofa.toml`; **Esc** cancels. Changes apply when starting
or resuming a session.

Detectors and improvement classification share the detector settings by default.
The improvement classifier suggests improvements after judging and is separate
from **Judge endpoint**. Optional overrides (`judge_model`, `judge_base_url`, and
`judge_api_key_env`) remain available in `autofa.toml`; existing overrides are preserved.

New configurations in this checkout use `openai/openai/gpt-6-luna` through NVIDIA for detectors and
the auxiliary improvement classifier. Existing explicit model choices, including
Gemini, are preserved. To switch an existing configuration, edit **Model** or
enter a model ID supported by the selected endpoint. Keep the full gateway model ID,
including its prefix. The discovery and behavior-judge models are independent.

**API preference** defaults to **Auto (prefer Decisions)**. The
[Decisions API](https://developers.openai.com/api/docs/guides/decisions)
returns fixed choices, predicate probabilities, and rubric scores. AutoFA checks the
configured endpoint, model and account, then uses `/decisions` for `classify_options`
and `llm_classify` with explicit options when supported. Generated text, extracted
quotes, improvement explanations and tool conversations use `/responses` when supported.
These outputs cannot be supplied by the Decisions API.

Population scoring does not automatically generate supporting quotes for each match.
It retains deterministic passages and defers optional LLM highlighting to final report
preparation. The reviewer can request supporting passages for an individual example
when needed. Generated highlights are cached and never determine a detector's label
or prove its causal explanation.

New detectors should use fixed-choice decisions for verdicts and subdecisions, without
generating a rationale or quote merely to justify every classification. Existing detectors
that extract a number, claim or passage and verify it in Python still need that extraction
to classify correctly. Moving that step to review would change their predictions; those
detectors must be revised and revalidated before removing it.

Select **Responses** to use `/responses` explicitly, or **Decisions (required for labels)**
to require native Decisions for finite-label calls. **Chat Completions** remains available
for compatible providers. Automatic routing falls back only on an explicitly unsupported
API or a request format that needs Chat Completions, keeping the same endpoint, model and
credentials. Authentication errors, rate limits,
outages and malformed responses are reported instead of being mistaken for absent support.
Startup prints the verified API selection after the ping, before discovery begins.

Capability checks are isolated by endpoint, model and account, and shared across worker
threads. The SDK's authenticated HTTP transport supports Decisions even when its installed
version has no generated `client.decisions` resource. Existing explicit API preferences
remain respected; change an old configuration's `feature_api` to `"auto"` to enable detection.

For explicit probability or score questions, detectors can call
`llm_client.decide(questions, text, focus="full")` with the documented question
objects. Answers retain request order; refusals remain explicit per-question
results. Native `decide` requires Decisions support and is unavailable with an explicit
Responses preference; AutoFA does not fabricate probability distributions from text.
`classify_options` raises on a refusal instead of returning a label.
Verified artifact images use inline data URLs, just as in the Chat path.

AutoFA's discovery instructions describe lexical, trace, semantic and composed
detectors. Startup guidance and `dataset_info.llm_detectors` expose the active
model/API and readiness. Discovery starts with an evidenced causal hypothesis, then chooses
deterministic code for explicit conditions or an LLM for semantic interpretation. Either
can be the first detector. Later discovery rounds use the **current library**:
`list_library_functions`, `pareto_status`, and `advance_round` expose counts of
deterministic and LLM detectors and suggest exploring the less represented
kind. Hybrids that call an LLM count as LLM detectors, including calls in pinned
helpers and dependencies. A supported causal hypothesis takes priority over these
suggestions; validation quality
determines which detectors are kept, with no required ratio. The same guidance
applies to both harnesses and discovery follow-up turns. LLM calls are recognized
for sampling, lint, strategy reporting, and checkpointing, including
`classify_options` and `decide`.

After each completed agent turn, current Good, Excellent and Outstanding LLM detectors
are evaluated on a representative 80% dataset panel and reviewed against the larger
match population. Current scans covering at least 80% and cached predictions are reused.
The agent receives qualification gaps and downgrade feedback before its next turn or
completion decision. This does not replace the fixed discovery comparison panel,
advance the round, or replace the final 100% validation. Interrupted checkpoints resume
their saved selection, including detectors whose scores already fell during validation.

LLM detectors and `llm_classify` accept model IDs served by an **OpenAI-compatible
Decisions, Responses or Chat Completions API**. Set its API root, model ID, and an environment
variable holding its bearer API key, or `"none"` to disable authentication.
Provider-specific SDKs and a provider registry
are not required. Native APIs with different protocols or authentication need a
compatible gateway.

For manual configuration, put the key in `~/.autofa/.env` (shared across worktrees),
the project's `.env`, or export it in your shell, then edit the
top-level fields in `autofa.toml` and restart AutoFA:

```bash
export LLM_API_KEY="your-api-key"
```

```toml
feature_base_url = "https://your-gateway.example/v1"
feature_api_key_env = "LLM_API_KEY"
feature_model = "your-provider/model-id"
judge_model = "your-provider/classifier-model-id"
```

`feature_*` configures LLM detectors and `llm_classify`. `judge_*` configures the
**improvement classifier**. Discovery and the behavior judge use their
selected Claude Code or Codex harness and its authentication.

The improvement classifier inherits the detector endpoint and key unless you set
`judge_base_url` and `judge_api_key_env`; its `judge_model` is always independent.
Choose model IDs supported by each endpoint. Changing endpoints does not translate
existing model IDs, so set the model IDs your endpoint serves.

Examples of API roots (documentation examples, not a built-in provider list):

| Service | API root | Typical key environment variable |
|---|---|---|
| [OpenRouter](https://openrouter.ai/docs/quickstart) | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| [Vercel AI Gateway](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions) | `https://ai-gateway.vercel.sh/v1` | `AI_GATEWAY_API_KEY` |
| [OpenAI](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create) | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| Local vLLM / SGLang | `http://localhost:8000/v1` (match your server's port) | `"none"` without authentication, otherwise your key variable |

Use the API root, including any path prefix such as `/api/v1`, without appending
`/chat/completions`. API keys stay in environment variables, never in the URL or
TOML file. When changing a role's endpoint, explicitly select its key environment
variable or choose `"none"`. Missing custom keys do not fall back to other credentials.

For a local server, first launch either
[vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) or
[SGLang](https://docs.sglang.io/docs/basic_usage/openai_api_completions)
in its installed environment. For example, with `MODEL_ID` set to your installed or
downloadable chat model, run **one** of:

```bash
vllm serve "$MODEL_ID" --host 127.0.0.1 --port 8000
# Or:
python -m sglang.launch_server --model-path "$MODEL_ID" --host 127.0.0.1 --port 8000
```

Once it is ready, configure AutoFA (replace both model IDs with the exact ID served
by the server; these are literal TOML strings, not shell variable substitutions):

```toml
feature_base_url = "http://localhost:8000/v1"
feature_api_key_env = "none"
feature_model = "your-served-model-id"
judge_base_url = ""      # inherit the detector server
judge_api_key_env = ""   # inherit its authentication setting
judge_model = "your-served-model-id"
```

`"none"` requires no key in your environment and never loads other provider
credentials; the client sends only a fixed `unused` bearer placeholder. If you launched
the server with an API key, use the name of an environment variable containing that
key instead. AutoFA connects through Chat Completions; you start and stop the model
server separately. Discovery and behavior judging use the selected Claude Code or
Codex harness independently of this detector endpoint.

Empty `feature_base_url` / `feature_api_key_env` inherit `hub_base_url` /
`hub_api_key_env`, which are blank unless a deployment preset sets them. With no
endpoint at all, LLM detectors and classifiers report that none is configured;
deterministic detectors still run. A preset key named `INFERENCE_HUB_API_KEY` also
checks `INFERENCE_API_KEY` as a fallback.

Optional request settings support model-specific token budgets, sampling, and
provider extensions. Discovery defaults are separate from shared classifier options:

```toml
# Discovery detector and improvement-classifier generation defaults:
feature_request_options = { reasoning_effort = "none" }
# Discovery improvement classifier only:
improvement_request_options = { reasoning_effort = "none" }
# Shared API classifiers, including taxonomy, retain endpoint defaults:
judge_request_options = {}
# To use the provider's default reasoning instead:
# feature_request_options = {}
# improvement_request_options = {}
# Example for a reasoning model that requires its default temperature:
# feature_request_options = { max_completion_tokens = 8192, omit_parameters = ["temperature"] }
# Preserve reasoning off when adding other request options:
# improvement_request_options = { reasoning_effort = "none", max_tokens = 4096 }
# Alternative provider controls (use the one your endpoint supports):
# feature_request_options = { extra_body = { reasoning = { enabled = false } } }
# feature_request_options = { extra_body = { chat_template_kwargs = { enable_thinking = false } } }
```

`reasoning_effort = "none"` is supported by compatible models through the
[OpenAI Chat API](https://developers.openai.com/api/docs/guides/reasoning) and
[OpenRouter's Chat API](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion).
OpenAI compatibility does not guarantee that every model can disable reasoning.
Use the endpoint's supported controls; AutoFA does not remove a rejected reasoning
setting or silently fall back to reasoning on. OpenRouter's `reasoning.exclude`
only hides reasoning text; it does not disable computation or its token cost.
Existing explicit option tables remain unchanged and replace the entire default
table, so add `reasoning_effort = "none"` to those tables to request it.
In particular, an existing `{}` means provider defaults, not reasoning off.
Explicit `judge_request_options` override the discovery improvement defaults.
Restart AutoFA for changed endpoint settings to take effect.

These settings override helper and raw detector Chat parameters, including the
startup probe. Without an explicit output-token limit, calls start with 64 tokens
for detector labels, 8 for the probe, and 1200 for
passage extraction. If one of these default budgets truncates a response, AutoFA
retries once with 4096 output tokens, allowing room for hidden reasoning. Explicit
token limits are never increased. Improvement classification starts with 2048 tokens.
`feature_request_options` also applies to passage extraction. Reasoning models may
need larger budgets because reasoning consumes output tokens. The client adapts
explicit API rejections
of `temperature` / `top_p` or requests to switch between `max_tokens` and
`max_completion_tokens`, without matching provider or model names. Set the options
explicitly to avoid those retries. Empty, refused, or truncated responses are
reported as errors rather than saved as labels.

Detector helpers read the final answer, ignoring separate reasoning fields and
removing leading `<think>`, `<thinking>`, or `<analysis>` blocks. Reasoning without
a final answer is an error. Use `llm_client.classify_options(...)` for a fixed set
of labels: it accepts a complete option, never a substring inside an explanation.
Reasoning cleanup preserves tags inside JSON evidence strings.

Request options apply to generated text through Responses or Chat Completions.
Fixed-label Decisions calls do not use generation options. The discovery agent and
behavior judge use their own harness settings. Discovery's `feature_api` and
reasoning-off defaults do not control taxonomy classification or the cheating check. Taxonomy
defaults to Chat Completions and accepts `--judge-param KEY=VALUE` (including
`api=responses`) and `judge_request_options` overrides. Cheating retains Chat
Completions and endpoint-default reasoning. Both retain their configured endpoint,
credentials, and model.
Options cannot replace evidence messages, the selected model, credentials, or
transport settings. Generated detectors should use `llm_client.classify_options(...)`
for labels or `llm_client.classify(...)` for free text; raw
`llm_client.chat.completions.create(...)` calls also inherit the configured options.
Chat-specific `with_raw_response`, `with_streaming_response`, `parse`, and `stream`
surfaces retain their Chat transport in automatic mode, including deferred requests.
An explicit Responses preference rejects requests it cannot represent faithfully.
Configured options take precedence, while unrelated call parameters and nested
provider settings are preserved. Changed detector options invalidate execution
checkpoints so predictions from a different reasoning setting are not reused.

### Permitted classifier alternatives

`autofa preflight classifiers` reads configured detector routes and credential-variable
presence without initializing settings or calling a model. The default alternatives are the
Luna medium, Sol medium and Sol high profiles described in
[configuration](configuration.md#optional-classifier-comparisons-during-discovery). An empty
alternative list means native discovery cannot escalate its classifier. Model access, vision and reasoning remain
unverified until checked; the harness model catalog does not establish access to detector
API deployments.

Preview an alternative selected for your provider and allowed spending scope:

```sh
autofa preflight classifiers --add-profile reasoning --model APPROVED_DEPLOYMENT \
  --base-url https://provider.example/v1 --key-env APPROVED_PROVIDER_KEY \
  --api chat_completions --request-options '{"reasoning_effort":"medium"}'
```

Replace the example deployment, endpoint and options with values supported by your provider;
use `--request-options '{}'` for provider defaults or `--key-env none` for an unauthenticated
local service. A changed provider requires an explicit credential variable. The preview
prints the complete proposed profile. Add `--write-config` to append it to future-run settings;
this does not change a running analysis or adopt a classifier. Existing names are rejected.
`--config PATH` can target an isolated campaign configuration.

`autofa preflight classifiers --probe reasoning` explicitly permits a bounded text/API
inference probe for that configured profile and may incur cost. It does not certify image
reading or reasoning quality. Use original artifact controls with positive, negative and
abstention expectations through native `classifier_profiles(compare)` to test those abilities.
The existing comparison/adoption workflow retains controls and budgets, preserves the judge
route, and requires fresh scoring after adoption. No additional model is enabled automatically.

Provider refusals and content filtering remain typed abstentions. Implicit short completions
can retry at 4096 tokens and use one final 8192-token rescue on truncation; explicit token limits
remain hard limits. A completed malformed short label can receive one formatting retry; AutoFA
never guesses a label from a misspelling. These bounded recovery calls consume normal usage.

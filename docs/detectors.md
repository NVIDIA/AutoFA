# Detectors and reusable helpers

[Back to README](../README.md) · [Metrics and feedback](metrics.md) · [Saved libraries](library.md)

[Concepts](#concepts) · [What you get](#what-you-get) · [Detector contract](#detector-contract) · [Repairing a judged detector](#repairing-a-judged-detector)

## Concepts

Three concepts describe AutoFA’s code and evaluation artifacts. See the
[glossary](glossary.md) for every other term.

| Concept | What it is |
|---------|------------|
| **Failure mode** | A recurring *why* the model fails — a behavioral pattern (e.g. “retries the same failing command instead of diagnosing the error”). Named with a punchy **description** and explained by a **mechanism** narrative. Not code. |
| **Detector** | Reusable Python code, `detect(item) → bool` (regex and/or LLM), that flags rollouts showing one failure mode, plus its metadata (`description`, `mechanism`, `function_mechanism`, scores). `submit_function` stores it in `library.json`. AutoFA scores detectors (precision, recall, `ppv@π₀`, `behavior_purity`) so you know which ones to keep, rewrite, or drop. The durable catalog is a library of detectors. |
| **Helper** | An ordinary reusable function, with an independent catalog and no detector scores. It can return parsed values, passages, or predicates for consuming helpers and detectors. |

## What you get

1. **Discovery** — sample traces, hypothesize failure mechanisms, write detectors
2. **Feedback** — each detector is scored immediately (keep / rewrite / drop)
3. **Judgment** — an independent Claude Code or Codex session checks whether labels match the claimed mechanism (`behavior_purity`)
4. **A durable catalog** — saved under `~/.autofa/library/…` so work survives across runs

Detectors inspect model content and recorded execution evidence, including tool results
and execution metadata. External task targets remain excluded. They can reuse other
detectors and ordinary unscored helpers; exported libraries carry their exact dependency
revisions so they also run outside the original session.

Detector `re` matching uses the `regex` engine with a default 0.1-second timeout.
Timed module functions and compiled-pattern methods keep the GIL by default so
other Python evaluation threads do not consume their process-CPU timeout while
matching. Explicit `concurrent` and `timeout` settings remain effective. This is
not per-thread CPU isolation: native work that releases the GIL, replacement
callbacks, and CPU consumed between iterator pulls can still affect the budget.

Reusable functions live in a separate **helper catalog**, persisted as `helpers.json`
and included in the portable `library.json` envelope. Helpers can parse tool calls,
normalize text, extract passages, or return predicates. They have no failure claim,
association score, judge, or promotion threshold. Import a standalone catalog with
`library_load(path=".../helpers.json")` to reuse it in another run. Only consuming detectors
are evaluated.

The discovery agent uses `list_helpers` / `get_helper` to find code, `submit_helper`
to save it, and `test_helper` to inspect outputs on example arguments or dataset
item IDs. `retain_as_helper` can preserve useful deterministic code from an active
or archived detector, independently of whether that detector was promoted. Helpers
can call other helpers. Helpers are deterministic functions with
bounded JSON results and optional input/output contracts; LLM decisions belong in
the consuming detector.

For example, submit a helper whose source uses
`autofa.trace.regex_passages(item, "reasoning_text", pattern)`, then submit a detector
with `uses_helpers={"candidates": "helper_name_or_revision"}`:

```python
def detect(item, llm_client=None):
    candidates = helpers.call("candidates", item)
    if candidates["status"] != "ok" or not candidates["complete"]:
        library.abstain("required evidence is unavailable or incomplete")
    if not candidates["value"]:
        return False
    evidence = helpers.render_passages(candidates, max_chars=12000)
    if not evidence["complete"]:
        library.abstain("candidate evidence exceeds the verifier budget")
    if llm_client is None or not getattr(llm_client, "available", True):
        library.abstain("LLM endpoint is unavailable")
    reply = llm_client.classify_options(
        evidence["text"], "Does the quoted evidence show the specific target behavior?",
        options=["yes", "no", "insufficient_evidence"],
        focus="full", max_chars=evidence["budget_chars"],
    )
    if reply == "insufficient_evidence":
        library.abstain("verifier could not decide")
    return reply == "yes"
```

Passage results include original field offsets, field hashes, matched text,
surrounding context, and explicit completeness. Abstentions are reported separately
from negative predictions, with coverage and conservative recall/FPR bounds.
`depends_on=["parent_name"]` with `library.fire("parent_name", item)` composes scored
detectors. Updating or deprecating a helper does not change existing consumers:
resubmit a detector to adopt the new revision, which invalidates its previous
execution and claim-review fingerprints.

## Detector contract

Agent trajectories are explored in execution order. Start with `get_trajectory`,
use `search_trajectory(item_id, query)` to search conversation steps (including long tails),
and open a hit with `get_trajectory_context(item_id, step_index, before=2, after=2)`.
Expand full contents with `get_trajectory_step` or `read_parsed_field`. Search and
context previews expose offsets, total lengths and explicit pagination/truncation.
Check surrounding tool results and later corrections before claiming an unresolved
failure. Detectors can use `autofa.trace.steps(item)` for ordered content, stable
`step_id`, optional API `source_event`, and `paired_step_index` / `result_step_indices`.
Combined text fields remain available for existing detectors and simple searches.

Use `get_trace_record(item_id)` to start with resource coverage, then select sections
such as `events`, `rollout`, `attempts`, `artifacts`, `tool_definitions`, `provenance`
or `relations`. Select a source event by `section="events", lane="", event_idx=259`;
its source index is distinct from the one-based parsed step index. JSON pointers and
`next_offset` expose all recorded values, including long text and future metadata
fields. `read_parsed_field(field="execution", pointer="/events/0/parts/0/text")`
provides the same nested access.

Bulk import does not fetch optional metadata separately for each rollout. When needed,
the analysis agent can call `get_trace_record(item_id, section="provenance", fetch_missing=True)`
(also `relations` or `prompt_summaries`). It reuses exported data or fetches and caches only
the requested resource. Repeat `fetch_missing=True` when paging supplemental results.
These results do not alter the dataset or detector inputs; ordinary reads remain offline.

`record` also retains exported `prompt_messages`, optional `prompt_summaries`,
and extension tables under `extra_resources`. Prompt-message pools and repeated
request context are captures, not additional actions. When source files supply
steps absent from the event API, `recovered_steps` exposes them separately with
their recorded metadata; it does not invent source event IDs.

Import warnings distinguish remaining evidence gaps (`Evidence limitation`) from
capture-format and context information (`Source note`). Request message/tool hashes
are resolved against the exported pools, while the original references stay compact.
Event request captures lack provider response envelopes by design; available raw
model-call responses supersede that note. Changes to recorded request context remain
available as separate snapshots and are not, by themselves, missing evidence.
Counts refer to occurrences or model requests, not failed tasks. New imports rebuild
the normalized cache after parser upgrades; existing saved runs are left unchanged.

In Python, `autofa.trace.record(item)` returns the full execution structure.
`events(item, lane=None)` includes all retained lanes; `metadata`, `attempts`,
`artifacts`, `tool_definitions`, `provenance`, `relations` and `coverage` provide typed
resource views. `Step.metadata` and `Step.parts` retain the source record alongside
the conversation view. These helpers have no text window. Resource availability
and coverage depend on the input; an empty unavailable resource
does not establish that no such event happened.

Source caches store duplicated step metadata, parts, and original call
fields as versioned references to the unchanged `execution` records. The normalized
step messages, observations, reasoning, tool names, and arguments remain inline for
detectors that read raw JSON. Use `autofa.trace.steps` for full step records;
the shared parser resolves their references without truncation. Imports reuse the
checked bulk download.

For bounded classifier input, `autofa.trace.step_evidence(item, pattern, max_chars=16000)`
scans the full ordered execution and selects matching steps, nearby context, linked tool
calls/results and closing steps. Large traces keep usable evidence instead of becoming
empty at a whole-trace character cutoff. Include the task requirement separately and align
the classifier's character budget with the rendered evidence. `selection_complete` reports
whether selected steps were rendered fully; `full_trace` is true only when every recorded
step is included without clipping. Omitted/truncated indices and source offsets support
follow-up reads with `steps`. Excerpts cannot establish absence of later repair or global
absence; retrieve the required context or leave that claim unknown. Helper submission and
inspection warn about length guards that discard all evidence, without changing their code.

Judge snapshots also replace exactly matching normalized text with source references.
Each distinct source event remains present, including auxiliary lanes and unique
metadata. Whole-trace verdicts still require reading the complete compact evidence;
reading a derived conversation view alone does not establish complete coverage.
Judge checkpoints require matching evidence-policy fingerprints. Invalidating a
checkpoint does not rewrite saved reports or scores.

Artifact-based tasks have a shared file-evidence workflow:

- `download_artifact(item_id, artifact_idx)` returns the verified cached original file.
- `read_artifact(item_id, artifact_idx)` reports an inspection manifest; use
  `section="segments"` and JSON pointers to page full located content.
- `view_artifact(item_id, artifact_idx, page=1)` returns an actual image block with
  an artifact hash, rendered-image hash and page/frame locator.

Text/CSV, DOCX paragraphs and tables, XLSX sheets/cells/formulas/cached values, PPTX
slides, and PDF pages retain stable locators. Image and PDF support uses Pillow and
PDFium, included in the normal installation and kept in sync by `uv run`. Native Office text
extraction works without Office software. Visual Office rendering requires
`libreoffice` or `soffice` on PATH and is a print-layout conversion: hidden content
may be omitted and formulas may recalculate. Inspect native cell evidence for the
original formulas/cached values. Macro-bearing or externally linked documents are
not visually rendered. Unsupported/encrypted files, absent renderers and resource
limits are reported explicitly, never treated as empty successful inspection.

MP4/QuickTime and WebM/Matroska video inspection uses PyAV, which AutoFA does not
install: build it against an FFmpeg on the host as described in
[audio, video and speech](artifact-dependencies.md#audio-video-and-speech). The manifest
reports stream metadata. `view_artifact(..., page=N)` displays the Nth decoded frame
of the first video stream, with its exact timestamp when recorded and a verified
frame locator. A selected frame can support a claim about its visible content;
it does not establish motion, intervening content, other streams or audio. Video
inspection does not issue whole-video completeness receipts. Decoding is limited
to 30 seconds per request, 18,000 frames and 16 megapixels per decoded frame;
unsupported media and exceeded limits remain explicit evidence gaps. Without PyAV, or
when the host FFmpeg lacks the file's demuxer or decoder (with the documented build,
H.264 and HEVC video in most MP4 and MOV files), the video is reported unavailable:
an evidence gap, never evidence that content is absent.

Readers can be checked on demand with
`prepare_artifact_support(capability="office" | "audio" | "speech")`. With
`install=True`, speech setup downloads an allowlisted local model (`tiny`, `base`, or
`small`) once PyAV and faster-whisper are installed. Office and audio setup only report
whether LibreOffice and PyAV are available; the tool never installs LibreOffice, PyAV,
faster-whisper or FFmpeg. Ordinary artifact reads and saved-evidence revalidation never
install packages or download models. See
[artifact dependency setup](artifact-dependencies.md) for platform requirements.

For ZIPs, inspect the member inventory, then pass `member_path="folder/file.pdf"`
to download/read/view/transcribe tools or the corresponding `trace` helpers. Only
that member is extracted. Receipts bind both archive and member hashes; listing
files does not inspect their contents. Nested, encrypted, or unsafe archives and
resource limits remain explicit; the original archive is still downloadable.

Audio inspection uses the same PyAV and is likewise reported unavailable without
it, or without the file's demuxer or decoder (with the documented build, `.mp3` and
`.aac` files and AAC audio in M4A files). `view_artifact(page=N)` shows the Nth
30-second mono waveform/spectrogram window with signal measurements. After speech setup,
`transcribe_artifact(start_seconds=0, end_seconds=30)` produces timestamped local
speech estimates; detectors use `trace.artifact_transcript` with the same window.
Original bytes, decoded samples, model assets and cached transcripts retain hashes.
Transcripts can be wrong. Signal plots and speech recognition do not establish
music content, tone, speaker identity, or perceptual audio quality; they are not
native listening. Use exact quoted transcript locators for speech citations and
leave unsupported acoustic claims unknown.

Detectors use the same cache and readers:

```python
from autofa import trace

def detect(item, llm_client=None):
    if llm_client is None or not getattr(llm_client, "available", True):
        library.abstain("LLM endpoint is unavailable")
    document = trace.artifact(item, 0)  # Select a recorded artifact_idx from trace.artifacts.
    image = trace.artifact_image(item, 0, page=1)
    if (document["status"] != "available" or not document["complete"]
            or image["status"] != "available"):
        library.abstain("required artifact evidence is unavailable")
    text = "\n".join(segment["text"] for segment in document["segments"])
    if not text.strip() or len(text) > 100_000:
        library.abstain("artifact text is unavailable or exceeds this detector's budget")
    verdict = llm_client.classify_options(
        text, "Does the chart contradict the accompanying values?",
        options=["yes", "no", "insufficient_evidence"],
        images=[image], focus="full", max_chars=100_000,
    )
    if verdict == "insufficient_evidence":
        library.abstain("the chart or its values cannot be assessed")
    return verdict == "yes"
```

Use a vision-capable detector model for `images`; image failures do not silently
retry without images. `llm_classify(artifact_images=[{"item_id": id,
"artifact_idx": idx, "page": 1}], ...)` prototypes the same input. Text windows
still apply, so select the relevant full segments and pages deliberately.
Judge sessions have artifact-specific readers, image views and citations tied to
exact source/transform hashes and locators. Saved detector results and judge reviews
revalidate the artifact evidence they actually used. Existing normalized caches need
reimporting with this adapter to acquire the trusted artifact source descriptor;
original bulk downloads can be reused without rerunning the evaluated model.

Metadata supports investigation, but token counts, input length, timing, or a failed
tool status alone do not establish an actionable behavioral mechanism. Discovery
must identify the decision and consequence, check recovery and counterexamples, and
explain what the mechanism adds beyond task difficulty or length. Submission lint
rejects known count/timing proxies without content evidence; behavioral review is
still necessary and cannot be replaced by static checks.

The `autofa.trace` API has explicit Python return contracts for steps, tool calls,
input messages and tool definitions, model calls, source events, parse results,
and passage evidence. `Step` remains a frozen dataclass; structured dictionary
results use `TypedDict` definitions and retain their existing keys and source
extensions. Optional keys and nullable values are distinguished. Provider-specific
payloads that cannot be described more narrowly remain explicitly opaque; annotations
do not add runtime validation or discard unsupported evidence.

Discovery receives an API reference generated from the actual signatures and types
in its system prompt. Each run also saves `trace_api.md`; the `get_trace_api` tool
returns bounded pages of the same reference, or a focused entry such as
`get_trace_api(name="steps")` or `get_trace_api(name="Step")`. Follow `next_offset`
with the same `name` until it is null, joining `reference` strings in order; check
that `reference_sha256` stays the same. These are the contracts of the Python helpers used
in detector code, separate from the MCP tools used to explore dataset items.
The reference includes defaults and evidence limits: `evidence_text` and
`text_window` default to 16,000 characters, including when `focus="full"`.

Local reports show a chronological, expandable event timeline with full-trace
search. Clicking a localisation passage opens its event and highlights the exact
Unicode character range. Evidence headers identify the local step, source viewer
event/part and lane, and the passage's line and character positions within that
step. Original detector-field coordinates remain separately labeled.
"Open viewer event" links to the corresponding EvalRDL viewer location, including
the event's lane. Missing event mappings are explicitly labeled as whole-rollout
links. The bundled local viewer can highlight passage text, including beyond collapsed
previews. Reports retain exact detector-field character offsets; ambiguous source
matches keep their event link without highlighting a different occurrence.
Source-only cloud reports use the saved event/part links without
embedding full traces. When localisation records need coordinate mapping,
AutoFA reads the affected examples to verify and map their coordinates;
it does not re-execute detectors or change the library.

Source adapters retain API part offsets. Event links require source provenance;
generic traces and raw-recovery steps never invent API
IDs from display positions. Stale or unmappable evidence retains its excerpt and
rollout link without claiming an exact event. Combined fields are collapsed in
the report but retain their original detector semantics.

Detector execution preserves complete available text, including derived reasoning,
actions, observations, and final messages. Smoke tests
sample rows without applying display truncation. Explicit LLM prompt windows and
UI/MCP previews remain bounded; select the relevant evidence before applying a
budget. Raw and parsed Gym imports preserve complete reasoning and responses;
parsed trajectories retain structured steps, user messages, and tool observations.

Gym imports reject malformed/unsupported records with file and line context.
They preserve repeat indices and unknown outcomes, and reject conflicting indices
or duplicate identities. Native Parquet indexes and default row streaming also
reject duplicate item IDs. Converted-data and identity-index cache versions prevent
reuse after conversion or indexing rules change.

`autofa.trace.evidence_text(..., focus="closing")` retains the actual ending of
the requested field; opening/closing windows select that field's content-bearing
steps. Character and step omissions are marked. A native field that differs from
the trajectory uses its own character window rather than being replaced by derived text.

To inspect the same fields detectors receive, use
`get_field_meta(item_id, field, view="detector")` and
`read_field_chunk(item_id, field, view="detector", offset=0)`. These return
provenance and stored/selected lengths. The default `view="stored"` reads the
canonical stored row; continuation hints preserve the chosen view.

Inputs and traces share a structural parser across imports, detectors, MCP tools,
and reports. It accepts native objects and JSON-encoded containers (up to eight
encoding layers), ATIF/steps envelopes, Chat Completions and Responses messages,
Anthropic content blocks, and NeMo Gym model-call captures. Parsing follows data
shapes rather than benchmark, harness or tool names. Literal message text and
strings inside tool parameters are preserved; arbitrary text is not recursively
treated as JSON. Unsupported structures and unavailable media remain explicit.

Detectors receive `input_messages`, `input_tools`, `input_text`, and `input_parse`
(origin and limitations). `autofa.trace` provides matching helpers, parsed
`tool_calls(item)` arguments and call IDs, `Step.observation_data`, and
`model_calls(item, offset=0, limit=1)` for recorded request/response context.
Calls made through a grader's model reference, such as an arena judge's verdict,
are excluded from items ([datasets](datasets.md)).
Original Parquet columns remain available; parsed tool arguments retain their raw
representation when decoding changes it. Invalid argument JSON remains text with
an `arguments_warning` instead of becoming an empty object.

MCP input inspection uses `get_input(item_id, section="messages"|"tools"|"text")`.
`get_model_calls(item_id, offset=0, limit=1)` expands captured request/response
references. `read_parsed_field(item_id, field="trajectory_steps",
pointer="/0/tool_calls/0/arguments")` navigates nested objects with JSON pointers;
it also supports `input_messages`, `input_tools`, `input_text`, `input_parse`,
and `model_calls`. Arrays/objects page by entries, strings by characters. Oversized
structures return child pointers instead of cutting JSON. These tools are
registered in the same MCP server used by discovery sessions.

Versioned source adapters and conversion caches rebuild newly selected inputs
when their normalization changes.
Recorded harness request snapshots are retained separately from executed steps,
so repeated context is never counted as repeated actions. A capture that starts
after model output is not presented as the initial request. Missing traces,
tool schemas, image inspection, or unrecognized formats are reported as limitations.

The read-only API audit can be rerun with
`.venv/bin/python tools/validate_input_parsing.py --run RUN_ID --output /tmp/input-audit`
(repeat `--run` for more runs). It checks source text and tool arguments against
the API, then checks detector, judge snapshot, and MCP views. Audit outputs stay in
the selected output directory.

The behavior judge opens a separate Claude Agent SDK or Codex session for each
sampled rollout. Codex judging exposes only the audited evidence tools, without
shell, file editing, or browsing tools. Each session receives a complete small input or an index of full content
fields, with paginated read/search tools for larger inputs. No character cap is
applied to the evidence archive or detector source. Known external outcome metadata
is excluded and those exclusions are recorded; model messages and tool arguments
are preserved. Missing content in the original input cannot be recovered by judging.

The default Codex harness uses GPT-6 Astra for discovery and GPT-6.1 Sol with medium
effort for behavior judging. With the Claude harness, the corresponding judge defaults are
`behavior_judge_model = "claude-sonnet-5-5"`,
`behavior_judge_thinking = "adaptive"`, and `behavior_judge_effort = "medium"`.
These settings apply to claim compilation, the compiler's coverage audit, each
initial review, and each independent confirmation. Discovery has its own `model`,
`thinking`, `effort`, and `thinking_budget_tokens`; improvement assessment keeps
`judge_model`.

For Claude in TOML, thinking accepts `default`, `adaptive`, `enabled` (budgeted), or
`disabled`. Effort accepts `default`, `low`, `medium`, `high`, `xhigh`, or `max`, subject to the
model's support. `default` delegates to Claude Code. Budgeted thinking uses
`behavior_judge_thinking_budget_tokens` (at least 1,024). The menu uses harness
capabilities to offer appropriate controls; Claude Code can normalize or ignore
unsupported combinations set manually. Effort is omitted for budgeted or disabled
thinking. Codex controls reasoning through `effort`, including any additional levels
reported by its model catalog (such as `ultra`); Claude thinking switches and token
budgets do not apply to Codex.

Changing the behavior judge harness, model, thinking mode, effort, or active token budget
invalidates previous judge checkpoints and requires fresh judgments. A processing
error retry must keep those settings; it cannot combine verdicts from different
judge configurations. Restart AutoFA after editing TOML to apply it to subsequent
judging.

Independent examples are judged in parallel, with `judge_workers = 4` by default.
Set an integer from 1 to 8 in `autofa.toml`; 1 restores serial judging. Claim compilation
runs once before the batch, and each positive example still receives its separate,
independent confirmation after its initial review. Results retain the original example
order. This reduces elapsed time without reducing the number of model calls; shared
rate limits and slow examples can limit the speedup. `llm_eval_workers` controls LLM
detector evaluations separately.

The judge limit applies to each batch, so simultaneous runs or batches multiply the
number of active sessions. After a connection failure, new examples stop starting and
in-flight stages can finish saving their checkpoints. Ctrl+C cancels active requests
and waits for SDK cleanup; completed stages remain saved. Resume reuses them even if
`judge_workers` changes. Restart AutoFA to load a changed worker setting; it does not
change an already running batch.

Failure localisation is collected when a detector positively labels an item, including
flagged examples that passed the external grader. `library.json` stores each detector's
`failure_localisations` keyed by item ID. Click a detector's bucket or example in the
HTML dashboard to see highlighted passages with surrounding words and their source field,
character offsets, and line/column coordinates. Clicking an individual example opens only
that item for the selected detector; clicking a bucket lists its examples. Context is
bounded to 12 words / 180 characters on each side, including languages without spaces.
Character offsets refer to the highlight, use Unicode code points, start at zero,
and exclude the end; lines and columns start at one. Derived-field coordinates refer to
the full detector field (`read_field_chunk(..., view="detector")`), not the dashboard's
pretty-printed trajectory. Localisation also travels with offline HTML/ZIP exports.
`get_library_function(name=..., item_id=...)` retrieves one example's evidence for a
specific detector; omitting `item_id` returns a preview of up to five records.

Regex `search`, `match`, `fullmatch`, `finditer`, and `findall` calls capture runtime
matches without changing the detector's return value; literal string membership rules
(`"text" in field`) and repeated fragments checked through dictionary counters or string
`.count()` are captured too. Exact field matches retain their field and occurrence;
quote-only matches prefer readable output over duplicated trace envelopes. Prompt parsing
hits are omitted when model-output evidence is available. LLM detectors keep their existing
classification calls; scoring does not automatically make extra requests for optional
highlights. Supporting passages are generated during final report preparation, or on an
explicit request from the reviewer for its current example. Completed highlights are reused.
Returned quotes must occur verbatim in content fields before coordinates are computed.
Each item retains up to eight passages of 600 characters; limits and extraction failures
are explicit. Absence/aggregate-count rules without repeated text and inputs that cannot be
mapped back to a field can have no local passage. Finalizing or explicitly saving the report
fills in pending highlights without rerunning detector classification.

The separate **behavior judge** receives these passages as navigation
hints alongside the full outcome-blind evidence. It must still read original context,
check counterevidence, and produce its own verified citations and verdict. Detector
quote extraction does not compute judge metrics or substitute for evidence reads.
The same localisation is available
to other judge consumers through the evidence snapshot, including improvement assessment.

Before reviewing any sampled rollout, a separate compiler session reads the original
description and mechanism and extracts their defining conditions. It sees no traces,
detector matches or grader outcomes. This happens once per detector judgment batch;
every sampled item receives the same frozen conditions. Each condition includes a
verbatim quote from the description or mechanism, a standalone predicate to assess, and
a minimum review scope (`local` or `whole_trace`).
A separate, trace-blind coverage review checks the draft against the original claim for
missing conditions, weakened predicates, insufficient review scope and defining claims
mislabeled as exclusions.
Rejected drafts receive bounded repair attempts; an unaudited draft cannot reach item
review. The final contract retains its coverage approval and compiler-policy fingerprint.
Contract version `behavior-claim-v4` retains the complete original claim alongside
these mandatory conditions and prerequisite, actor, causal-link, outcome/recovery and
counterevidence checks. Failed compilation produces processing errors, never a weaker
fallback contract. The judge assesses every requirement as `supported`,
`contradicted` or `unknown`, with verified supporting and opposing citations.
The application derives each review's verdict: any defining contradiction rejects the claim;
otherwise any unknown prevents confirmation; only all-supported confirms it.
Editorial calibration cannot override these decisions. The judge-only trajectory
view preserves customer messages and tool callers so simulator actions and later
recoveries remain visible. Contract checks retain the original alternatives,
examples and quantifiers; they do not invent additional clauses in the literal claim.

Separately, every contract includes a mandatory `failure_contribution` acceptance check:
the detected behavior must explain a root cause of the task failure. Reviewers must
identify an unmet task requirement, independent evidence of the final defect or incompletion,
and the causal connection, considering later recovery and competing explanations. A recovered
parsing error followed by an unrelated incorrect artifact is rejected even if every literal
claim about the parsing error is true. Wasted turns, a failure label or statistical association
alone are insufficient; missing ending or causal evidence remains uncertain. The behavior need
not be the sole cause, but a lasting consequence such as budget exhaustion preventing required
work needs direct evidence. Narrowing the detector's description cannot waive this check.
Earlier decisions can be root causes; the last visible error need not be one. A record
stopping at an intermediate error alone does not prove task termination. Generic doubt,
missing verification or submission is insufficient without a concrete decision or omission
connected to the actual defect. The judge must not invent an unobserved cognitive cause.
An observed violation in the submitted artifact can establish the defect without hidden
grader results. A passing local test does not excuse violating a different explicit
requirement. This does not waive the claim's other defining conditions.

Discovery receives this same acceptance requirement before authoring detectors. Its candidate
check connects the requirement, specific action, resulting defect and recovery evidence.
It asks for the mistaken rule or operation, its closest competing explanation, and a
targeted corrective action, clearly marked as a proposal. A shared causal family may have
different technical subtypes and remedies, provided every reviewed positive establishes
the common causal connection. Bare labels such as
"fabrication" or "hang" are search leads; a useful card identifies the evidenced causal
decision. Discovery compares its predicate with the closest available hard negative,
including the same operation permitted by a different task requirement. Mentioning an
object or action does not establish whether the task requires or forbids it.
An attempted repair is not evidence of successful recovery. Error-only matchers belong in
unscored helpers; missing causal evidence calls for investigation or abstention. Each batch
may contain fewer than `top_x` supported candidates while exploration continues toward the
configured minimum. New, seeded and resumed sessions use the same guidance, and claim repair
cannot replace a missing causal explanation with an occurrence-only description.

`smoke_test_feature` can evaluate exact `item_ids` with `expected_outputs` and
`expected_abstentions`, returning each prediction and any mismatches or processing errors.
Explicit IDs override random sampling; missing IDs fail before detector execution. Discovery
uses these controls before submission to test its claimed distinction, recovery and missing
evidence handling. Author-selected expectations test the implementation; they are not
independent validation or evidence of corpus purity. Random smoke sampling remains available.

A provisional confirmation receives an independent confirmation check in a fresh
review session. Neither reviewer sees the detector's assigned label, and the second
reviewer receives no first-review reasoning, assessments or citations. It counts as
confirmed only if that check also supports the full
claim and its contribution to the task failure. A rejection, uncertainty, or processing error
in that check becomes the final result. Initial reviews remain available as provisional evidence; each item
contributes only its final result to the counts and purity denominator.

Native automatic compaction stays enabled for both harnesses. Persistent review
notes and read coverage survive compaction, the frozen contract remains available through `review_status`,
and substantive assessments must cite exact passages read since the
latest compaction. Each supported clause with a frozen `whole_trace` scope requires full
stored-field coverage, even if another clause remains unknown or contradicted. A reviewer
cannot bypass this by choosing a local review. A local counterexample can refute an
exhaustive claim without establishing all remaining clauses. Citation
validation establishes what was read, not whether the judge's interpretation is
correct. Missing semantic evidence produces `uncertain`; incomplete sessions and
invalid outputs produce separate processing errors. Schema/reference errors get
up to two repair attempts within the original budget, while valid uncertain
assessments are not retried to manufacture a score. Both unknowns and errors remain
in the sampled denominator. A batch without any confirmed or rejected claims has
no purity/specificity score. Claim validation has no shortened-evidence or heuristic
purity fallback. Detector specificity describes how much technical detail the claim isolates. A broad
causal family can be valid without specifying one identical low-level mistake across all
examples. Vacuous outcomes and structural proxies still fail causal attribution. Missing or invalid
specificity remains unavailable; code complexity and LLM usage
cannot supply a replacement score. An unavailable specificity score preserves the claim
verdict but cannot satisfy an objective requiring that metric.
Saving or publishing preserves these results. Retry processing errors through an
explicit repair plan during discovery; publishing does not reopen reviews or repair plans.

Reports display confirmed, rejected, uncertain and processing-error counts with
per-item requirements, including the compiled conditions and original quoted wording.
Changed claims or detector source invalidate their judgments.
A stable claim fingerprint identifies the original wording,
detector source and contract version. The compiled contract has a separate fingerprint
that also covers its defining conditions. Current-contract scores require a valid
claim fingerprint and matching static judge-policy fingerprint, covering the compiler,
both review prompts and evidence/claim versions. Missing or changed fingerprints
invalidate display and ranking even when the literal claim is unchanged. The recorded judge model and
full run-configuration fingerprint remain available as provenance; historical runs
are not invalidated merely because global model settings change. Strong benchmark
association is displayed separately from full claim validation. A numerical purity score is
the share of sampled fires with both the claim and its contribution to task failure confirmed,
not proof of an unseen grader outcome or validation of every match. Reports identify whether
each displayed example was actually reviewed. Scores from earlier contracts must be re-judged
before they can validate root causes; the version and policy changes invalidate cached reviews.
Compare frontier scores only when their claim contracts and judging policies match.
Reusing a compiled contract requires a matching compiler-policy fingerprint. Its model
and source-quoted exclusions remain frozen in the contract, and new compilations retain
their session usage and costs with the result.

Versioned evidence snapshots, prompts, replies, verdicts, saved notes, read ranges,
compaction events and session usage are stored under
`behavior_judge/<detector>-<unique ID>/<item index>/`. Latest-result files remain
available at `behavior_judge/<detector>.{evidence,verdict}.json`; previous batches
are retained. Each initial review has a 64-turn, 10-minute budget. A provisional
confirmation receives a separate check with its own 64-turn, 10-minute budget;
rejected and uncertain initial reviews do not receive that additional check.

Optional semantic regression checks exercise claim extraction and the real judge on
synthetic controls. They use the existing Claude login and require an explicit model:

```bash
AUTOFA_RUN_LIVE_JUDGE=1 AUTOFA_LIVE_JUDGE_MODEL=YOUR_CLAUDE_MODEL_ID \
  pytest -q tests/test_claim_judge_live.py
```

Ordinary test runs skip these checks. Live evidence, replies, verdicts, and judge
provenance remain in pytest's temporary directory; the controls contain no private
dataset records or grader outcomes.

Use the sandbox-safe `autofa.trace` API to read normalized traces, rather than
selecting raw `chain_of_thought` / `response_text` fields yourself. For example,
this deterministic candidate finder checks the final assistant message for a
test-success claim. Validate that candidate against tool results before treating
it as a failure mode:

```python
def detect(item, llm_client=None):
    import re
    from autofa import trace

    message = trace.final_message(item)
    if not message.strip():
        library.abstain("final assistant message is unavailable")
    return bool(re.search(r"\ball tests passed\b|\bpytest\b[^\n]*\bpassed\b", message, re.I))
```

For a semantic check, read the ordered messages, tool calls, and observations with
`trace.actions`. Use fixed answer options, return a boolean, and abstain when the
endpoint or required evidence is unavailable. This example retains the complete
available action text within an explicit budget instead of silently slicing it:

```python
def detect(item, llm_client=None):
    from autofa import trace

    if llm_client is None or not getattr(llm_client, "available", True):
        library.abstain("LLM endpoint is unavailable")
    text = trace.actions(item)
    budget = 32000
    if not text.strip() or len(text) > budget:
        library.abstain("action evidence is unavailable or exceeds this detector's budget")
    verdict = llm_client.classify_options(
        text,
        "Does the final assistant claim the tests passed despite the latest relevant "
        "test output showing failures, with no later successful retest? Check tool "
        "results and subsequent corrections. Choose insufficient_evidence if the "
        "required test results or final response are missing or ambiguous.",
        options=["yes", "no", "insufficient_evidence"],
        focus="full", max_chars=budget,
    )
    if verdict == "insufficient_evidence":
        library.abstain("the trace does not establish the required evidence")
    return verdict == "yes"
```

For finer selection, use `trace.steps(item)` and its typed `Step` attributes,
or `trace.regex_passages` / `trace.render_passages` with their completeness flags.
`trace.coverage(item)` describes upstream gaps; a complete text window does not
prove that every source event was recorded. Inspect that coverage before making
whole-trace or absence claims. `library.abstain(reason)` is supplied by the detector
sandbox and records an abstention separately from positive and negative labels.

Allowed imports include `re`, `json`, `math`, `autofa.trace` (also
`from autofa import trace`), and `autofa.llm.compat`.

Each submit should include a punchy **description** (failure-mode title for the card
headline, e.g. “Writes its own test, sees it pass, ships on it”), a 2–4 sentence
**mechanism** — an evidence-backed narrative (observed decision → supported cause →
observed consequence, with hypotheses and visibility limits identified), and a
**function_mechanism** — fluent prose for
what the detector code does (fields, regex/LLM, when it returns true). Avoid
detector-trigger prose like “Fires when…” for the failure mechanism. Claims must be
**backed by detector evidence**: after eval, inspect `example_fail_ids` / `behavior_purity`
and align the claim with what the detector actually labels (or retune it).

## Repairing a judged detector

Use the MCP repair workflow before editing a judged detector:

1. Call `get_function_repair(name)` for supported conditions, contradictions,
   missing facts, processing errors, and concrete evidence-inspection calls.
   Examples are bounded; every finding includes its full matching item IDs.
   Retrieve full assessments with `get_function_feedback(name, offset=0, limit=10)`
   and follow `pagination.next_offset`, or select `item_id`.
2. Call `plan_function_repair(name, action, rationale, finding_ids, evidence_ids)`
   using the returned finding IDs and their matching evidence IDs. Available
   actions depend on the findings:

   | Action | Purpose |
   | --- | --- |
   | `tighten_detector` | Change the detector while preserving the written claim. |
   | `revise_claim` | State a narrower, supported claim; unknown causes are not facts. |
   | `split_detector` | Propose distinct children with their own validation. |
   | `repair_implementation` | Fix recorded execution errors while preserving the written claim and detector relationships. |
   | `gather_evidence` | Investigate missing or ambiguous facts without assuming false hits. |
   | `retry_review` | Retry recorded processing errors for the unchanged detector and judge policy. |
   | `retire` | Explicitly retire an unsupported detector while preserving its history. |

3. For an edit, call `submit_function(..., repair_plan_id=plan_id)` to evaluate the changed
   candidate, then follow `review_schedule` and `grade_gap` for development review or
   a repair for the actual limiting dimension. Already frozen repair reviews keep their cohort
   and recovery path.
   Keep `evaluate_functions(names=[name], mode="full")` for the runtime's final-validation
   handoff. To save an edit for later validation, use `auto_evaluate=False`,
   then explicitly evaluate and call `judge_function(name)` when ready.
   Link split children with the same `repair_plan_id`. Evidence gathering and
   processing retries retain the current version and use `judge_function(name)` explicitly;
   retirement uses `remove_function`.
   Use `defer_function_repair(name, reason)` when the required evidence is unavailable.

During discovery, a deferred automatic review is an instruction to follow its actionable
diagnosis. Specializing to a coherent subtype can improve precision even at lower coverage;
revise the claim consistently, preserve supported originals, and avoid task-ID exceptions.
When precision is from 40% to below 70%, deliberately use
`judge_function(name, review_mode="diagnostic")` for up to three examples if per-case claim
evidence is needed to choose the repair, or `review_mode="standard"` to validate a useful OK
detector. Diagnostic findings do not produce a main adherence score or confirmed support.
Explicit `judge_function(name)` still defaults to standard review.
See [review scheduling](metrics.md) for the thresholds and unchanged hard prefilters.

After an unproductive edit or quality regression, including a loss within the same grade,
call `list_function_versions(name)`. Read its comparisons and recommendation, and the retained
lessons from `get_function_repair(name)`:
compare compatible score scopes, lost supported matches and the configured trade-off.
Different cohorts do not establish a measured regression. If an earlier supported version is
preferable, explicitly defer any active repair and call
`restore_function_version(name, version_hash, reason, expected_current_version_hash)` with the
returned hashes and an evidence-backed reason. The recommendation never restores automatically.

Restoration preserves what earlier attempts learned; it does not reset repair budgets or review
exposure, rerun evaluation, or create fresh validation. Recheck each retained lesson against the
restored source and claim. A broader claim's counterexample may not apply to an earlier narrow
claim, but restoration cannot erase applicable counterevidence. Retained lessons remain historical
investigation evidence, separate from current verdicts; their history references preserve the
full edits, findings and evidence for later inspection.
Before another edit, inspect the restore result's suggested bounded smoke check and the old
claim's evidence. `smoke_test_feature(name=..., expected_version_hash=..., item_ids=...)`
checks the exact restored version and its frozen dependencies, reusing compatible cached results
or performing bounded diagnostic work. If it still fires, inspect whether the cited contradiction
applies to the restored claim. A nonfire only excludes that observed match; an abstention or error
remains unresolved. None of these smoke outcomes establishes causal validity, and restoration
itself starts no evaluation or paid review.

The run's `detector_repairs.json` retains the original baseline, before/after
versions, judgments, evidence reservations, and outcomes. The limit is **three
repair attempts per detector**, including deferred attempts, and survives Resume.
Processing retries do not consume this budget and remain available after it is exhausted.
Implementation repairs do consume it. They can proceed even when the behavior review
passed, using current recorded error cases as evidence. Smoke-test those cases and
supported controls, then reevaluate and review the changed code; old confirmations do
not transfer. For transient provider faults, restore the service and retry evaluation.
An unreserved submitted repair that still fails execution must be deferred before another
bounded edit; an already reserved review keeps its original recovery path.
Evidence projection changes invalidate judgments without deleting their history;
those claims need a current review before contributing to validated rankings.


A deferred candidate with complete, current causal support can request ordinary independent
confirmation without inventing another repair. The submitted version, dependencies, claim,
judge policy and settings must still match; an unfinished reservation or recorded repair outcome
cannot be bypassed. This also applies after the edit budget is exhausted, without granting
another edit. The prior repair remains deferred or exhausted with its history unchanged.
Family prerequisites, task exposure and confirmation budgets still apply, and development
reviews never become fresh confirmation. Blocked review/repair responses and
`get_function_repair` provide a `next_action` with an executable tool call and an explanation
of the prerequisite; resolve that prerequisite before retrying. Interrupted ordinary
confirmation retains its original requested sample size and task cohort.

Library files are replaced atomically, but the repair ledger and library are separate
writes. If a valid review is recorded and the library save then fails, the tool reports
the error, keeps the previous library scores, and retains the reviewed candidate in the
ledger's `after_validation` snapshot. Recovery is explicit; failed writes do not
automatically replay reviews or refund repair attempts.

New repair validation excludes recorded review/repair items and previously reserved
validation items. **This is not a guaranteed unseen holdout:** other exposure through
the discovery agent is not tracked. After extending the permitted scan, a smaller
fresh sample is still judged. Scores retain the target and actual sample sizes and
remain outside the validated ranking when the target is unmet. If no fresh matches
remain, AutoFA re-judges previously reviewed matches for the current detector and
marks the scores as reused evidence; this does not validate the repair. Processing
retries replace only failed rows, preserve the frozen claim contract and prior
semantic judgments, and keep the original sample denominator and evidence limits.

Publication builds the report from saved results without starting or retrying reviews.
Report cards require computed precision, recall, normalized precision, adherence, and
behavioral specificity. Candidates without these scores remain in the saved library
and contribute to the report's unavailable-score count. This includes completed
inconclusive reviews; a missing value does not mean a review is still running.
Raw precision ("Failure rate when flagged") and its failed/flagged counts appear in
the detector details.

A completed plan does not automatically promote a detector. Check confirmed,
rejected, uncertain, and processing-error counts. Revised wording receives **new
claim validation**; it does not retroactively confirm the original claim. Pure
uncertainty calls for evidence gathering, while cited support for part of the claim
can also justify proposing a narrower claim for fresh evaluation.

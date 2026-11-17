# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public dictionary contracts for :mod:`autofa.trace`.

These are typing and documentation contracts, not runtime validators. Helpers
return ordinary dictionaries and retain provider extensions where the parser
already preserves them. ``RawValue`` marks evidence that is deliberately not
validated: check its shape before indexing it. Recognized content/function
schemas describe the common forms, with a raw alternative for other values.
"""

from typing import Annotated, Literal, NotRequired, Required, TypedDict

__all__ = [
    "RawValue", "ContentBlock", "MessageContent", "Message", "ToolFunction",
    "ToolDefinition", "ToolCall", "LocatedToolCall", "SourceEvent", "ModelCall",
    "ToolCount", "ParseSuccess", "ParseFailure", "ParseSummary", "Passage",
    "PassageExtraction", "RenderedPassages",
    "ExecutionMetadata", "TracePart", "TraceEvent", "SourceToolCall", "TraceAttempt",
    "TraceArtifact", "RecordedToolDefinition", "TraceProvenance", "TraceRelation",
    "ResourceCoverage", "TraceCoverage", "TraceRecord", "ModelCallMetadata",
    "EventRequestText", "PromptSummary", "RecordedPromptMessage",
    "ArtifactSegment", "ArtifactEvidence", "ArtifactImage", "ArtifactTranscript",
    "SpeechSegment", "SpeechModel", "SpeechAudio",
]

type RawValue = Annotated[
    object,
    "Opaque provider data preserved without schema validation; narrow with isinstance/key checks.",
]


class ContentBlock(TypedDict, total=False):
    """Recognized provider content block; keys depend on its type.

    Images, audio, encrypted reasoning and unknown blocks are preserved evidence,
    not claims that AutoFA inspected their contents. Malformed/other block shapes
    remain RawValue; use isinstance and key checks before reading them.
    """

    type: Annotated[str, "Provider block kind, e.g. text, image, thinking or tool_result."]
    text: Annotated[str, "Text of a recognized text/input_text/output_text block."]
    thinking: Annotated[str, "Readable thinking block content when recorded."]
    id: Annotated[RawValue, "Provider block or tool-call identity, when recorded."]
    name: Annotated[str, "Tool name for a recognized tool_use block."]
    input: Annotated[RawValue, "Tool-use arguments; may be an object, list, scalar or raw text."]
    content: Annotated[RawValue, "Nested tool-result or provider-specific content."]
    tool_use_id: Annotated[RawValue, "Identity of the call producing this tool result."]
    is_error: Annotated[bool, "Provider tool-result error flag, when recorded."]
    image_url: Annotated[RawValue, "Image reference; image contents are not inspected here."]
    source: Annotated[RawValue, "Provider media source; may contain references or encoded data."]
    data: Annotated[RawValue, "Opaque block payload, including redacted thinking."]


type MessageContent = str | list[ContentBlock | RawValue] | ContentBlock | RawValue


class ToolFunction(TypedDict, total=False):
    """Common nested function descriptor; provider fields are not schema-validated.

    Used inside calls and definitions. Its name and schema are normally strings
    and objects respectively, but their original values remain available.
    """

    name: Annotated[RawValue, "Recorded function name (normally a string)."]
    description: Annotated[RawValue, "Recorded description (normally a string)."]
    arguments: Annotated[RawValue, "Decoded object/array or unchanged scalar/raw arguments."]
    arguments_raw: Annotated[RawValue, "Original arguments when decoding changed the value."]
    parameters: Annotated[RawValue, "Recorded parameter schema; decode is not validation."]
    parameters_raw: Annotated[RawValue, "Original parameter schema when decoding changed it."]
    input_schema: Annotated[RawValue, "Alternative provider parameter schema."]
    input_schema_raw: Annotated[RawValue, "Original input schema when decoding changed it."]
    strict: Annotated[RawValue, "Provider strict-schema flag, when recorded."]


class ToolDefinition(ToolFunction, total=False):
    """Recorded initial or per-request tool definition; no keys are guaranteed.

    Supports flat and nested function schemas and provider-native tools. Unknown
    definition fields remain in the dictionary. An empty tools list means no
    usable definitions were recorded, not proof that no tools were available.
    """

    type: Annotated[RawValue, "Provider definition kind, normally a string."]
    function: Annotated[ToolFunction | RawValue, "Nested function descriptor, if present."]
    function_raw: Annotated[RawValue, "Original nested descriptor when decoding changed it."]


class ToolCall(TypedDict):
    """Normalized call with guaranteed name/arguments and optional recorded identity.

    Object/array argument strings are decoded; scalar strings and malformed JSON
    remain unchanged. Raw evidence and extra provider fields are retained.
    """

    name: Annotated[str, "Canonical tool name; 'unknown' if no usable name was recorded."]
    arguments: Annotated[RawValue, "Decoded object/array, scalar, raw text, or None."]
    call_id: NotRequired[Annotated[RawValue, "Recorded call identity; not assumed to be a string."]]
    id: NotRequired[Annotated[RawValue, "Original provider identity, when present."]]
    tool_call_id: NotRequired[Annotated[RawValue, "Original alternative call identity."]]
    type: NotRequired[Annotated[RawValue, "Original provider call kind."]]
    function: NotRequired[Annotated[ToolFunction | RawValue, "Original nested call descriptor."]]
    function_raw: NotRequired[Annotated[RawValue, "Nested descriptor before structural decoding."]]
    arguments_raw: NotRequired[Annotated[RawValue, "Arguments before structural decoding."]]
    arguments_warning: NotRequired[Annotated[
        RawValue, "Usually a string explaining retained invalid JSON; source values can persist.",
    ]]
    source_call_index: NotRequired[Annotated[RawValue, "Provider call position within its event."]]
    result_event_idx: NotRequired[Annotated[RawValue, "Recorded external result-event identity."]]


class LocatedToolCall(ToolCall):
    """A ToolCall with its 1-based originating position in steps(item)."""

    step: Annotated[int, "1-based parsed step index; not an external viewer event ID."]


class Message(TypedDict, total=False):
    """Structured message returned by input_messages or expanded model_calls.

    A role is always supplied, but other fields depend on the provider. Most roles
    are strings; a malformed role in a recognized provider record can survive.
    Content may be text, blocks, null or unsupported raw data. Missing content is
    normal for function-call/reasoning records. Provider extensions are retained
    after evaluation-data filtering; unsupported message records remain in the
    original source rather than being invented as readable messages.
    """

    role: Required[Annotated[RawValue, "Role, normally user/assistant/system/developer/tool."]]
    content: Annotated[MessageContent, "Text or content blocks; unsupported values stay raw."]
    type: Annotated[RawValue, "Provider record kind, e.g. message/function_call/reasoning."]
    id: Annotated[RawValue, "Recorded message identity."]
    name: Annotated[RawValue, "Recorded author or function name."]
    text: Annotated[RawValue, "Alternative provider text field."]
    reasoning_content: Annotated[MessageContent, "Recorded reasoning, if exposed by the source."]
    reasoning: Annotated[MessageContent, "Alternative reasoning content field."]
    summary: Annotated[MessageContent, "Reasoning summary blocks; not necessarily full reasoning."]
    tool_calls: Annotated[
        list[ToolCall] | RawValue, "Normalized calls, or retained malformed data.",
    ]
    function_call: Annotated[ToolCall | RawValue, "Legacy function call or retained raw value."]
    call_id: Annotated[RawValue, "Call identity on Responses-style call/result records."]
    tool_call_id: Annotated[RawValue, "Call identity on Chat Completions tool results."]
    tool_use_id: Annotated[RawValue, "Call identity on Anthropic tool results."]
    arguments: Annotated[RawValue, "Decoded call arguments, or unchanged scalar/raw data."]
    output: Annotated[RawValue, "Recorded function-call output; containers may be decoded."]
    input: Annotated[RawValue, "Provider tool-use input."]
    is_error: Annotated[RawValue, "Recorded provider error flag."]
    refusal: Annotated[RawValue, "Recorded refusal content."]
    annotations: Annotated[RawValue, "Recorded provider annotations."]
    audio: Annotated[RawValue, "Opaque audio data or reference."]
    encrypted_content: Annotated[RawValue, "Opaque encrypted content, not readable reasoning."]
    content_raw: Annotated[RawValue, "Original content when structural decoding changed it."]
    arguments_raw: Annotated[RawValue, "Original arguments when structural decoding changed them."]
    arguments_warning: Annotated[RawValue, "Explanation of arguments retained without decoding."]
    output_raw: Annotated[RawValue, "Original function output when decoding changed it."]


class SourceEvent(TypedDict):
    """Verified source API event identity; not the 1-based parsed step position.

    Available only for original API event exports with compatible provenance.
    Local/raw-recovered trajectories do not invent viewer identities.
    """

    event_idx: Annotated[int, "Nonnegative source event ID used in viewer e<id> links."]
    lane: Annotated[RawValue, "Source lane, normally a string (empty for conversation lane)."]
    part_idx: NotRequired[Annotated[int, "Source part ID when exact passage mapping provides it."]]


class ModelCall(TypedDict):
    """Expanded recorded model request/response, separate from execution steps.

    Context can repeat prior assistant actions: never replay these snapshots as
    a trajectory. Optional source metadata is copied without schema validation;
    absent recorded flags do not imply True. Unknown call metadata is retained.
    """

    request: Annotated[list[Message], "Expanded request messages; [] when absent."]
    response: Annotated[list[Message], "Expanded response messages; [] when absent."]
    tools: Annotated[list[ToolDefinition], "Expanded recorded tool definitions; [] when absent."]
    index: NotRequired[Annotated[RawValue, "Recorded 0-based model-call index, when available."]]
    request_recorded: NotRequired[Annotated[RawValue, "Normally bool: was a request captured?"]]
    response_recorded: NotRequired[Annotated[RawValue, "Normally bool: was a response captured?"]]
    initial_request_recorded: NotRequired[Annotated[
        RawValue, "Normally bool: is this snapshot known to precede model output?",
    ]]
    source_event_index: NotRequired[Annotated[RawValue, "Event containing the request snapshot."]]
    source_lane: NotRequired[Annotated[RawValue, "Recorded lane of the request-snapshot event."]]
    limitations: NotRequired[Annotated[
        RawValue, "Normally list[str] of parse/source limitations; absent when none recorded.",
    ]]
    metadata: NotRequired["ModelCallMetadata"]
    request_metadata: NotRequired["ModelCallMetadata"]
    response_metadata: NotRequired["ModelCallMetadata"]


class ModelCallMetadata(TypedDict, total=False):
    """Recorded metadata excluding message bodies; unknown provider fields survive.

    Usually request metadata contains model/temperature/max_tokens and response
    metadata contains usage/finish information. Missing keys mean not recorded;
    values are source evidence rather than a provider-normalized common schema.
    """

    model: RawValue
    model_name: RawValue
    parameters: RawValue
    temperature: RawValue
    top_p: RawValue
    max_tokens: RawValue
    max_output_tokens: RawValue
    seed: RawValue
    usage: RawValue
    finish_reason: RawValue
    stop_reason: RawValue
    status: RawValue
    choices: RawValue
    latency_s: RawValue
    request_dialect: RawValue
    request_params_raw: RawValue


class ExecutionMetadata(TypedDict, total=False):
    """Rollout/event metadata with optional original fields and open extensions.

    All values remain source data: timestamps are usually strings, token counts
    integers, latency_s seconds. No status or finish reason is inferred. Evaluation
    outcomes are excluded during import. Narrow RawValue before arithmetic/indexing.
    """

    run_id: RawValue
    task_id: RawValue
    repeat: RawValue
    agent_ref: RawValue
    record_schema: RawValue
    status: RawValue
    error: RawValue
    started_at: RawValue
    finished_at: RawValue
    ended_at: RawValue
    latency_s: RawValue
    latency_ttft_ms: RawValue
    model_name: RawValue
    finish_reason: RawValue
    prompt_tokens: RawValue
    completion_tokens: RawValue
    total_tokens: RawValue
    cached_tokens: RawValue
    reasoning_tokens: RawValue
    event_idx: Annotated[RawValue, "Original source identity when recorded; never a parsed index."]
    source_event_id: RawValue
    kind: RawValue
    role: RawValue
    actor: RawValue
    lane: Annotated[RawValue, "Source lane; '' is the conversation lane, missing is unknown."]
    lane_kind: RawValue
    request_idx: RawValue
    parent_event_id: RawValue
    paired_event_idx: RawValue
    paired_call_idx: RawValue
    length_truncated: RawValue
    has_encrypted_reasoning: RawValue
    is_copied_context: RawValue
    n_events: RawValue
    n_tool_calls: RawValue
    n_artifacts: RawValue
    raw_record_ref: RawValue
    src_path: RawValue
    src_line: RawValue


class TracePart(TypedDict, total=False):
    """Original event part, including full text when retrieved and opaque media metadata.

    Metadata-only/image/encrypted parts are not claims of inspected content.
    Missing text differs from an explicitly recorded empty string.
    """

    part_idx: RawValue
    part_type: RawValue
    channel: RawValue
    origin: RawValue
    text: RawValue
    chars: RawValue
    tokens: RawValue
    media_type: RawValue
    artifact_idx: RawValue
    event_idx: RawValue
    lane: RawValue


class TraceEvent(ExecutionMetadata, total=False):
    """Original source event with parts; arbitrary event kinds and imported lanes remain.

    Legacy fallback returns parsed step records instead: check coverage()['events'].
    In that fallback step_index is 1-based and event_idx is never manufactured.
    """

    parts: list[TracePart] | RawValue
    step_index: int
    source: RawValue
    step_id: RawValue
    message: RawValue
    reasoning_content: RawValue
    observation: RawValue
    tool_calls: list[ToolCall] | RawValue
    metadata: ExecutionMetadata | RawValue
    event_text: "EventRequestText | RawValue"


class EventRequestText(TypedDict, total=False):
    """Full original request snapshot fields and parameters, when recorded by the source."""

    request_dialect: RawValue
    request_messages: RawValue
    request_tools: RawValue
    request_params: RawValue


class PromptSummary(TypedDict, total=False):
    """Viewer prompt-history metadata; kept/dropped/added summaries are not complete messages."""

    base_event_idx: RawValue
    event_idx: RawValue
    lane: RawValue
    kept: RawValue
    dropped: RawValue
    added: RawValue


class RecordedPromptMessage(TypedDict, total=False):
    """Captured prompt-message pool row; msg_json retains the recorded message payload."""

    msg_idx: RawValue
    msg_hash: RawValue
    msg_json: RawValue


class SourceToolCall(ExecutionMetadata, total=False):
    """Original tool-call row with full arguments, status, timings and pairing when recorded.

    Legacy fallback contains normalized calls with a 1-based step field and partial
    coverage. args_head alone is a preview, not the complete arguments.
    """

    call_idx: RawValue
    tool_name: RawValue
    call_id_source: RawValue
    args_full: RawValue
    args_chars: RawValue
    args_head: RawValue
    result_event_idx: RawValue
    step: int
    name: str
    arguments: RawValue


class TraceAttempt(TypedDict, total=False):
    """Recorded execution attempt, including error/termination information when captured."""

    attempt_idx: RawValue
    failure_class: RawValue
    is_terminal: RawValue
    error_message: RawValue
    started_at: RawValue
    ended_at: RawValue


class TraceArtifact(TypedDict, total=False):
    """Artifact metadata and optional retrieved text; helpers never fetch a URI themselves."""

    artifact_idx: RawValue
    name: RawValue
    kind: RawValue
    role: RawValue
    origin: RawValue
    uri: RawValue
    mime: RawValue
    size_bytes: RawValue
    sha256: RawValue
    truncated: RawValue
    completeness: RawValue
    event_idx: RawValue
    call_idx: RawValue
    part_idx: RawValue
    text: RawValue


class ArtifactSegment(TypedDict):
    """Full extracted text with a stable page/sheet/cell/slide/paragraph locator."""

    locator: str
    text: str
    sha256: NotRequired[Annotated[str, "Hash of the extracted segment text, not the source file."]]
    location: NotRequired[dict[str, RawValue]]
    formula: NotRequired[RawValue]
    cached_value: NotRequired[RawValue]


class ArtifactEvidence(TypedDict, total=False):
    """On-demand artifact inspection. Check status/complete/limitations before deciding."""

    status: Literal["available", "partial", "unavailable", "unsupported", "error"]
    artifact_idx: int
    sha256: str
    size_bytes: int
    metadata: TraceArtifact
    source_identity: dict[str, RawValue]
    download_path: str
    format: str
    mime_type: str
    transform_version: str
    segments: list[ArtifactSegment]
    complete: bool
    visual_complete: bool
    requires_visual_review: bool
    visual_pages: int | None
    scope: str
    verified: bool
    limitations: list[str]
    reason: str


class ArtifactImage(TypedDict, total=False):
    """One rendered page. data_url carries actual image bytes for classify(images=[...])."""

    status: Literal["available", "partial", "unavailable", "unsupported", "error"]
    artifact_idx: int
    artifact_sha256: str | None
    sha256: str
    source_identity: dict[str, RawValue]
    page: int
    page_count: int
    width: int
    height: int
    locator: str
    mime_type: str
    transform_version: str
    data_url: str
    complete: bool
    limitations: list[str]
    reason: str


class SpeechSegment(ArtifactSegment):
    """Machine-estimated speech with absolute timestamps in the original audio."""

    start_seconds: float
    end_seconds: float


class SpeechModel(TypedDict):
    """Pinned local speech-model identity; private asset paths are not exposed."""

    status: Literal["available"]
    name: str
    sha256: str
    engine: str
    engine_version: str
    device: str
    compute_type: str


class SpeechAudio(TypedDict, total=False):
    """Decoded time window and its provenance; not an assessment of audio quality."""

    source_sha256: str
    stream_index: int
    audio_streams: int
    duration_seconds: float | None
    sample_rate: int
    channels: int
    codec: str
    window_seconds: int
    declared_window_count: int | None
    start_seconds: float
    end_seconds: float
    decoded_sample_rate: int
    decoded_samples: int
    decoded_pcm_sha256: str
    peak_amplitude: float
    rms_amplitude: float
    locator: str
    audio_complete: Literal[False]


class ArtifactTranscript(TypedDict, total=False):
    """Pinned speech estimate. Missing/failed reads retain empty segments and a reason.

    Neither a transcript nor an empty speech segment proves absence of other audio,
    speaker identity, sound quality, or complete inspection of the source artifact.
    """

    status: Literal["available", "unavailable", "unsupported", "error"]
    artifact_idx: int
    sha256: str
    source_sha256: str
    size_bytes: int
    metadata: TraceArtifact
    source_identity: dict[str, RawValue]
    path: Annotated[str, "Local path to verified original or archive-member bytes."]
    verified: Annotated[
        bool, "Verifies source bytes, not speech accuracy or inspection completeness.",
    ]
    cache_key: str
    member_path: str
    archive_sha256: str
    transform_version: str
    speech_transform_version: str
    segments: list[SpeechSegment]
    audio: SpeechAudio
    model: SpeechModel
    language: str
    complete: Literal[False]
    audio_complete: Literal[False]
    scope: str
    limitations: list[str]
    reason: str


class RecordedToolDefinition(TypedDict, total=False):
    """Source definition metadata with optional full def_json; not just initial input_tools."""

    def_idx: RawValue
    name: RawValue
    tool_type: RawValue
    def_chars: RawValue
    sha256: RawValue
    offered_by: RawValue
    def_json: RawValue


class TraceProvenance(TypedDict, total=False):
    """Recorded source location/reader information; retained extensions are not inferred."""

    path: RawValue
    file: RawValue
    first_line: RawValue
    last_line: RawValue
    reader: RawValue
    producer: RawValue
    site: RawValue
    input: RawValue


class TraceRelation(TypedDict, total=False):
    """Recorded relation between source entities, distinct from inferred call/result pairing."""

    space: RawValue
    kind: RawValue
    frm_key: RawValue
    frm_lane: RawValue
    frm_task_id: RawValue
    to_key: RawValue
    to_lane: RawValue
    to_task_id: RawValue
    resolved: RawValue
    reason: RawValue
    asserted_by: RawValue
    asserted_at: RawValue


class ResourceCoverage(TypedDict):
    """Availability of one imported resource, independent of whether an agent reviewed it.

    available with [] can mean recorded empty; unavailable with [] means unknown.
    partial means a projection, omissions or missing text; inspect reason and any
    retained importer extensions for the exact scope.
    """

    status: Literal["available", "unavailable", "partial"]
    reason: NotRequired[str]


type TraceCoverage = dict[str, ResourceCoverage]


class TraceRecord(TypedDict):
    """Complete imported execution view; helpers add no default character budget.

    Every listed resource key is present. Check availability before treating an
    empty collection as evidence of absence. Legacy exports provide only partial
    parsed events/calls; original metadata/resources are not invented. Additional
    importer keys are retained. Returned containers are independent deep copies.
    """

    rollout: ExecutionMetadata
    events: list[TraceEvent]
    tool_calls: list[SourceToolCall]
    attempts: list[TraceAttempt]
    artifacts: list[TraceArtifact]
    tool_definitions: list[RecordedToolDefinition]
    provenance: list[TraceProvenance]
    relations: list[TraceRelation]
    availability: TraceCoverage
    unparsed_resources: NotRequired[Annotated[
        dict[str, RawValue], "Original resource values with an unsupported collection shape.",
    ]]
    raw_trajectory_metadata: NotRequired[Annotated[
        dict[str, RawValue], "Source trajectory envelope metadata, excluding evaluation outcomes.",
    ]]
    prompt_summaries: NotRequired[list[PromptSummary]]
    prompt_messages: NotRequired[list[RecordedPromptMessage]]
    extra_resources: NotRequired[Annotated[
        dict[str, list[dict[str, RawValue]]],
        "Additional imported rows, keyed by source table name.",
    ]]
    resource_metadata: NotRequired[Annotated[
        dict[str, dict[str, RawValue]], "Original resource response metadata, keyed by resource.",
    ]]
    recovered_steps: NotRequired[Annotated[
        list[TraceEvent], "Recovered raw execution, separate from API events; see coverage.",
    ]]
    recovered_steps_source: NotRequired[Annotated[
        RawValue, "Storage reference expanded by record(); 'steps' refers to the envelope's steps.",
    ]]


class ToolCount(TypedDict):
    """Tool name and number of calls in a successfully parsed trajectory."""

    name: str
    count: int


class ParseSuccess(TypedDict):
    """Successful trajectory summary; obtain event contents with steps(item)."""

    ok: Literal[True]
    dialect: str | None
    n_steps: int
    n_agent_steps: int
    n_tool_calls: int
    tools: list[ToolCount]
    agent_name: str | None
    limitations: Annotated[
        list[str], "Known parse omissions; success is not proof of completeness.",
    ]


class ParseFailure(TypedDict):
    """No usable trajectory. Empty input has no warning key; failed parses may have one."""

    ok: Literal[False]
    warning: NotRequired[str | None]


type ParseSummary = ParseSuccess | ParseFailure


class Passage(TypedDict):
    """Exact regex match and context in one original field, not reserialized JSON."""

    field: str
    start: Annotated[int, "0-based Unicode character offset, inclusive."]
    end: Annotated[int, "0-based Unicode character offset, exclusive."]
    quote: Annotated[str, "Original field[start:end]."]
    context: Annotated[str, "Original field[context_start:context_end]."]
    context_start: Annotated[int, "0-based Unicode character offset, inclusive."]
    context_end: Annotated[int, "0-based Unicode character offset, exclusive."]
    field_chars: Annotated[int, "Length of the original field in Unicode characters."]
    start_line: Annotated[int, "1-based start line in the original field."]
    start_column: Annotated[int, "1-based start column in the original field."]
    end_line: Annotated[int, "1-based exclusive-end line in the original field."]
    end_column: Annotated[int, "1-based exclusive-end column in the original field."]
    field_sha256: Annotated[str, "SHA256 of the full original field encoded as UTF-8."]
    method: Annotated[str, "Extraction method; regex_passages uses regex_extractor."]


class PassageExtraction(TypedDict):
    """Regex extraction result; no match is distinct from unavailable/incomplete evidence.

    status='ok', complete=True, value=[] means no nonempty matches. Missing or
    non-string fields use status='unavailable', complete=False. Match caps produce
    status='ok', complete=False. Disallowed fields and regex errors raise instead.
    """

    status: Literal["ok", "unavailable"]
    complete: Annotated[bool, "Whether all nonempty matches in the requested field were returned."]
    value: list[Passage]
    reason: Annotated[str, "Empty on complete extraction; otherwise explains unavailability/cap."]


class RenderedPassages(TypedDict):
    """Whole passage contexts fitting a character budget; no partial context is emitted."""

    text: str
    complete: Annotated[
        bool, "False if extraction was incomplete/unavailable or any context omitted.",
    ]
    budget_chars: Annotated[int, "Effective character budget, clamped to [64, 100000]."]


class StepEvidence(TypedDict):
    """Selected readable step content, excluding attachments and unrepresented parts.

    Selection/regex absence cannot establish semantic absence or no later repair.
    Indices are 1-based positions in steps(item), not external viewer identities.
    """

    status: Literal["ok", "unavailable"]
    text: str
    full_trace: Annotated[
        bool, "All parsed steps' textual projections rendered wholly; source may be partial.",
    ]
    selection_complete: Annotated[bool, "All selected steps' textual projections rendered wholly."]
    total_steps: int
    matched_step_indices: Annotated[list[int], "All regex-matching steps."]
    selected_step_indices: list[int]
    omitted_step_indices: Annotated[list[int], "All steps not rendered, selected or otherwise."]
    truncated_step_indices: Annotated[list[int], "Rendered only as excerpts."]
    budget_chars: Annotated[int, "Effective text budget, clamped to [256, 100000]."]
    reason: str


class CommandFragment(TypedDict):
    """Complete command field, or the exact prefix of the field containing the anchor."""

    step_index: Annotated[int, "1-based position in steps(item)."]
    path: Annotated[str, "Command location within that step, e.g. message.commands[0].keystrokes."]
    text: Annotated[str, "Original decoded command text; the anchor itself is excluded."]


class CommandAnchor(TypedDict):
    """Unique exact quote in a decoded command field; offsets are Unicode characters."""

    step_index: int
    path: str
    start: Annotated[int, "0-based inclusive offset in the command field."]
    end: Annotated[int, "0-based exclusive offset in the command field."]
    quote: str


class CommandPrefix(TypedDict):
    """Unavailable prefixes cannot establish absence of a prior command."""

    status: Literal["ok", "unavailable"]
    complete: bool
    commands: list[CommandFragment]
    anchor: CommandAnchor | None
    reason: str

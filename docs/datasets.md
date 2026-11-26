# Datasets and rollout selection

[Back to README](../README.md) · [Local imports and traces](local-traces.md) · [Batch runs](workflows.md#batch-runs)

## Dataset wizard

Arrow-key navigation throughout:

The home screen shows the selected model and library connection status. Config
paths, memory details, detector fields and objectives are available under
**Settings → System information**.

- **Source** — **Local eval logs (NeMo Gym / Harbor)** or **EvalRDL remote server**
  (shown when a remote server address is configured)
- **Filesystem browser** — **↑/↓**, **Enter** to open, **←/Backspace** up; use a folder or pick a `.parquet` / `.jsonl`. Recognized NeMo Gym / Harbor folders show the recognition message directly above **[ USE THIS FOLDER ]**, as one highlighted menu item. Recognition uses explicit framework configuration or matching log signatures and supports single-job launcher folders. Empty configured runs are recognized; generic metadata and mixed download folders are not sufficient.
- **Supported sources**
  - **Local NeMo Gym and Harbor runs** — select the complete run folder, retaining
    its configuration, logs, and companion files. Recognized runs are imported into
    immutable EvalRDL snapshots under `~/.autofa/sources/`; see
    [local imports](local-traces.md).
  - **EvalRDL remote server** — browse one newest-first table containing model, benchmark, harness, rollout count, user, date, name, and ID; download `GET /v1/runs/{run_id}/export/autofa` into an immutable generation under `~/.autofa/converted/remote/`, keyed by API origin and canonical run ID
  - **Remote filters** — type plain text such as `super` to search every column, or narrow columns with `model:super`, `benchmark:swebench`, `harness:nemo-gym`, `user:alice`, `name:candidate`, `date:2026-09`, `rollouts:2500`, or `id:abc`
  - **Remote multi-select** — **Tab/Enter** moves from the filter to the table; use **↑/↓** to navigate, **Space** to check rollout sets, **a** to toggle all matches, and **Enter** to download the checked rows. At least one row must be checked. Selections persist across searches, and the footer summarizes their benchmarks, including hidden selections. **/** returns to the filter and **←/→** reveals off-screen columns. All models and benchmarks in the checked rollout sets are included without a second filter prompt.
  - **Remote sorting** — press **s** in the table to sort by a column and **r** to sort **↑/↓**. The active column shows **↑** or **↓**; missing values remain last.
  - **Compatibility inputs for failure discovery** — existing AutoFA / leaderboard
    Parquet and standalone trajectory JSONL remain supported. Generic JSONL uses
    `~/.autofa/converted/`. Disable efficiency metrics for these inputs; they do
    not provide the canonical EvalRDL tables that efficiency metrics require.
- **Local-file benchmarks** — **multi-select** (**Enter** toggles, `[ DONE ]` at top finishes, **a** = all) then confirm “You selected … Is this OK?”
- **Discovery depth** — interactive launches choose 1 / 3 / 5 / 8 rounds, or type a positive number in Custom rounds. After dataset preparation, a visible 60-second countdown selects the currently highlighted option (initially the configured `max_rounds`, In-depth with 5 rounds by default). Arrow keys change the selection without stopping the countdown; Enter selects immediately and Esc cancels. Custom entry and back navigation do not restart the timer. `--no-auto` disables it. Non-interactive New/Continue uses the configured depth and, when discovery is enabled, exits after the discovery turn.
- **Minimum Good detectors** — saved in Settings, default **1**, with no extra prompt per run. Existing saved choices and `--min-failure-modes` overrides are preserved. Reaching the minimum can require extra rounds beyond the selected depth.
- **Runtime estimates** — show a median and observed range only with enough measured rounds from matching model/settings runs and similar log size/depth. With limited history, a clearly labeled rough range projects measured round durations and identifies uncalibrated models/settings or workloads. A timing guide explains the uncertainty above the choices, and hour-based estimates omit minute suffixes. During discovery, **Round N of M · ETA** refreshes each second with minutes remaining, an approximate finish time and an uncertainty range. After two measured rounds, the central estimate follows the latest three rounds’ median pace. Partial turns and idle time do not count as completed rounds. Extra rounds to meet the minimum are excluded. A pricing disclaimer explains that cost depends on model rates, token usage and extra rounds; cost figures are omitted. See [estimate calibration](estimate-calibration.md) for methodology and the audit of prior logs.
- **Eval safety** — `submit_function` / default evaluation uses the development policy for
  fresh partitioned runs, and the existing capped probe otherwise (`eval_limit=5000`,
  configurable). Dataset summaries stream counts and text-length statistics without caching full traces. Detector errors and abstentions retain their diagnostic type and message, without retaining traceback frames and completed input batches.
- **Chunk sizing** — on launch AutoFA probes available RAM (Linux `MemAvailable`, macOS `vm_stat`) and estimates row memory from uncompressed Parquet metadata with Python overhead. Automatic sizing rounds down and permits one-row batches for large traces or low memory. Set `eval_chunk_size = 512` (or any >0) in `autofa.toml` to pin a row limit; `0` = auto. Readers can return smaller batches to keep their estimated and actual byte usage bounded.
- **Scoring throughput** — full, sampled, and judge evidence scans feed a persistent worker pool across read batches. Small Parquet batches do not serialize LLM detectors. Each queue bounds pending work by job count and estimated resident row bytes (shared rows count once), with at most one lookahead row; oversized rows remain intact. Results stay in source order for reproducible scores and review sampling. Selected-row scans reuse file readers and metadata, releasing them on completion or interruption.
- **Full corpus** — the agent can inspect with `dataset_chunks` / `peek_chunk`, then call `evaluate_functions(mode="full")` to stream-score every matching row. Byte limits control batching, not evidence truncation: a single oversized trace is still read intact and can exceed the batch budget. These limits are not a process-wide memory ceiling for parsers, concurrent workers, or external harnesses.
- Agent samples via MCP (small probes + optional chunk peeks), not a giant dump into context

Combined datasets have a stable name such as `mixture-7f42e183c952a610` in the CLI,
catalog path, report, and cloud library. The name hashes the source set and model/
benchmark filters, independent of selection order. Remote sources use their Data
API origin and run ID, so cache locations and trace-adapter versions do not change
the name; local sources use a full file-content hash. The name is saved as
`dataset_id` in the run config, separately from the model filter. Single-source,
single-model runs retain their model name.

**Remote shortcut (non-interactive download):**

```bash
# Download your run without immediately starting discovery
autofa new --api-run RUN_UUID --no-auto --eval-limit 200

# Optional API base override
autofa new --api-run <run_uuid> --api-base https://eval.example.org

# Fetch updated exports and rebuild the trace cache, even for a previously downloaded run
autofa new --api-run <run_uuid> --refresh-api
```

`--api-run` takes the EvalRDL run uuid, the 32-hex address in the run's
`#/run/<run_uuid>` link; the run's 16-hex `run_id` also works. AutoFA filters the
trace tables by `run_uuid` (by `run_id` on a server whose trace tables have no
`run_uuid` column) and links reports to `#/run/<run_uuid>`.

API docs: [EvalRDL remote server OpenAPI](https://eval.example.org/api/docs). Catalog filters: `GET /v1/runs?benchmark=swebench-verified`.

### Choosing rollouts

`autofa prepare` creates an analysis-ready, reusable rollout selection. Remote runs
enter through the EvalRDL API; local NeMo Gym / Harbor logs are converted through
the same EvalRDL data model. Run it without arguments to browse local data or search
the remote catalog by model, benchmark, date, and other fields, then select exact runs.
When `--input` names a parent directory containing multiple declared Gym or Harbor
runs, AutoFA discovers them recursively and imports each run separately.

```bash
# Interactive picker
autofa prepare

# One or more known runs
autofa prepare --api-run <run_uuid>

# The five newest matching catalog runs
autofa prepare --model <model> --benchmark <benchmark> --latest 5

# Ten failed rollouts from local logs
autofa prepare --input ./gym-run --reward '<=0' --per-run 10

# Reuse a saved selection
autofa prepare --input <selection_id>

# Select accepted taxonomy failures for discovery
autofa prepare --input ./taxonomy-results --group accepted --label OWN-IGM
```

The interactive **New analysis** flow offers the same choice under **Dataset source >
Taxonomy classification results**: pick a results file or folder, then the judge
view, group, labels, and evidence support.

Taxonomy-group discovery treats every selected classification as a failure and
pairs it with up to one discovery-success control from the same original run.
Canonical EvalRDL controls use the same `reward >= 0.5` correctness boundary as
the discovery export. AutoFA reports when it loads controls from the original
source. For taxonomy input, `--limit N` caps the selected failures across runs
before controls are added, so the prepared dataset can contain up to `2N` rows.

Repeat `--api-run`, `--input`, `--model`, `--benchmark`, `--task`, or `--reward`
to include multiple values. Reward comparisons combine with AND, so
`--reward '>=0' --reward '<1'` selects partial scores. `--per-run N` limits each
run. For evaluation inputs, `--limit N` limits the combined selection; for
taxonomy input it limits failures before controls. Add `--sample random --seed
N` for a reproducible random sample. Catalog-wide `--model`, `--benchmark`, and
`--runs-where` queries require `--latest N`.

For advanced remote queries, `--runs-where` and `--where` pass EvalRDL expressions
unchanged to `di.runs` and `di.rollouts` and combine with the named filters using
`AND`. Use EvalRDL SQL syntax and single-quoted strings, for example
`--where "status = 'ok' AND reward < 0.5"`. Local imports do not include that
query engine. `--dry-run` reports selection counts without saving or downloading.

**Workflow boundaries**

- `autofa new --input ...` or `--api-run ...` starts discovery.
- `autofa prepare` is optional preparation for workflows that reuse an exact selection.
- `autofa classify` assigns failures to a fixed taxonomy in a separate workflow.

Remote runs load original event text and full tool arguments by default. AutoFA
downloads `GET /v1/runs/{run_id}/export/trace`, then joins its Parquet tables locally
using rollout, agent, conversation lane, event, and part/call identities. A confirmed
complete run uses one export. Subsets, including runs whose total rollout count is
unknown, use explicit selections of at most 500 rollouts per request. Completed
batches are reused on retry.
The compact `/export/autofa` Parquet supplies labels and input text, with a separate
checked source cache and `manifest.json`. Large trace tables are indexed on disk
so importing a run does not load all of its text into memory. Grader/verdict lanes
remain excluded from detector-visible trajectories, with the request messages and
tools that only their events sent.

The trace cache retains non-grading lanes, original event/part records, tool execution
status, timing, token usage, provider extensions, attempts, artifact metadata and
exported text, tool definitions, provenance and relations. The selected conversation
still supplies the ordered `steps` view; auxiliary lanes stay separate. Known target
fields and transport credentials are removed at metadata boundaries; model-visible
content, tool arguments/results and schemas retain their original payloads.
Missing provenance, relations and prompt-history summaries can be filled through
cached optional API reads. Prompt summaries are not full request messages;
captured requests come from event text or original source records.

The compact export preserves uncapped reasoning and includes `system_text`
and the selected conversation `lane`. System instructions are available to detectors
and evidence review; derived inputs keep system and user messages separate. The
compact export omits intermediate messages and tool observations, so it does
not replace the full trace export. `--refresh-api` forces a new download and trace reconstruction
for New/Continue when an upstream run changes.
Each refresh writes a new export generation and publishes it only after its dataset
and provenance are complete. Saved runs keep their original paths and bytes; failed
refreshes leave the previous export usable. A run uuid, its `run_id` and other
aliases resolved by the catalog share the canonical `run_id`'s cache.
For Responses-API/NeMo Gym sources, original JSONL records supply recorded model
requests and responses omitted by the normalized event API. Empty event traces
are reconstructed from these model calls, including reasoning and tool calls.
Available tool results are recovered from request messages; missing requests or
results are reported explicitly, never invented. Later request snapshots and
tool definitions remain available in `response_text.model_call_evidence`, with
indexed message/tool pools that avoid duplicating repeated conversation history.
Existing event sequences remain authoritative when present. Request/response
metadata is also preserved, excluding external targets and transport credentials.
Calls made through a grader's model reference (`model_ref`, at the top level or in
`response_metadata`, whose name contains the word `judge`, `grader`, `scorer`,
`evaluator` or `genrm`, or is `reward_model`; for example `judge_model`, `genrm_model`
or `aalcr_v1_1_judge_model`) are not execution evidence. A reference containing the
word `policy` is never a grader's. An arena judge's comparison and verdict, for example,
are excluded from steps, inputs and `model_call_evidence` and counted in
`evidence_scope.limitations`.
The event export does not record whose call a `model_call` event was, only its
`model_call_id` (`source_event_id`): events with a grader's call ID go too, with
their tool calls and the request messages and tools only they sent. Local NeMo Gym
imports read the references from each rollout's raw record the same way. A grader
that reuses the policy's own model reference cannot be told apart.
Raw files are downloaded once with declared-size/checksum checks. JSONL records
and model calls are parsed incrementally, with no fixed file or record size cutoff.
Byte offsets are read locally without lossy
UTF-8 window joins; malformed or truncated records still stop preparation. Source
versions are separate, so existing runs keep their original input unchanged. A source
cache prepared before adapter version 15 can hold graders' calls, so AutoFA refuses
to load its rows; start a new analysis to re-prepare such a source.
Incomplete archives, mismatched counts, or unavailable full text stop preparation
rather than silently starting a summary-only analysis. Completed bulk downloads and
local indexes are reused after interruption; interrupted downloads restart atomically.
The import indicator distinguishes trace-table and raw-log downloads, showing bytes,
transfer speed, and an ETA when the total size is known. Verification, indexing, and
rollout preparation have their own status messages. A raw log can be much larger than
its trace archive, and its progress starts from zero for each file.
Raw-log connection drops, incomplete responses, and transient HTTP errors retry
automatically with bounded backoff (six attempts by default). The indicator announces
each retry and resets its byte count, speed, and ETA. Raw transfers restart from zero:
the source endpoint may not support byte-range resume. Only complete, size-checked
downloads enter the reusable cache; retries keep already completed trace archives.
Invalid ZIPs, corrupt Parquet payloads, and inconsistent manifests or joins invalidate
the affected download so retry can fetch a repaired archive. Local disk or index-write
failures retain the downloaded archive for retry.
Exported artifact text is available immediately. Retained artifacts can also be
downloaded on demand through the configured source API, including externally stored
files that its download endpoint resolves. Cached original bytes are verified against
declared sizes/hashes and pinned across discovery, detector evaluation and judging.
Arbitrary URLs and local paths inside trace content are not download instructions.
Encrypted reasoning remains unavailable. Downloaded files and extracted text do not
establish visual review; use the artifact image tools for that. Benchmark
outcome labels remain those supplied by the API, including
its reward threshold; verify their meaning before interpreting failure metrics.
`remote_full_traces = false` in `autofa.toml` explicitly selects the limited compact
export instead, with a visibility warning. It is not suitable for validating harness
behavior. Locally selected compact API exports use the same source recovery.
Standard `<run>_autofa.parquet` and `<run>_<task>_<repeat>_autofa.parquet` filenames
provide source identity; renamed exports need a `<filename>.source.json` sidecar
(replacing `.parquet`) containing `run_id` and `base_url`, or the original
`source.json` alongside the export. The API origin must match `data_api_base_url`.
Flat files with the exact compact-export schema and no provenance stop with
instructions; `remote_full_traces=false` explicitly permits summary-only loading.
Other local Parquets and structured Gym traces remain local.

Declared compact exports treat `system_text`, `prompt_text`, and generated output as
literal text, even when that text contains JSON resembling messages or requests.
Recorded input captures remain authoritative where available. Input views retain
separate system and developer messages and remove only exact duplicate system text.
Detectors and judge excerpts use the same recovered input view. Raw JSONL recovery
rejects inconsistent task and repeat identities before attaching a request to a rollout.

Saved runs keep their bound dataset. Use `--refresh-api` with New/Continue to
download and normalize remote data again. Missing data in a standalone normalized
file must be recovered from its original source. Inspect
`autofa.trace.coverage(item)` for resource-specific `available`, `partial`, or
`unavailable` status, including upstream truncation and absent tables.

Expected parquet columns (also accepted via aliases in parentheses):

| Kind | Fields |
|------|--------|
| Identity | `model_id`, `benchmark`, `problem_id` (`task_id`), `run_idx` |
| Labels (never readable by detectors) | `is_correct` (`resolved` / `tb_is_resolved`) / `accuracy` |
| Content (default allowlist) | `system_text`, `prompt_text` (`instruction` / `question`), `response_text` (`generation`), `chain_of_thought` / `reasoning_text`, … |

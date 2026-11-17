# Running an analysis

[Back to README](../README.md) · [Inputs](datasets.md) · [Configuration](configuration.md) · [Saved results](library.md)

[Batch runs](#batch-runs) · [How a session feels](#how-a-session-feels) · [Discovery limits and continued campaigns](#discovery-limits-and-continued-campaigns) · [Efficiency metrics](#efficiency-metrics) · [Taxonomy classification](#taxonomy-classification) · [Preflight checks](#preflight-checks) · [Who does what](#who-does-what) · [Tips](#tips)

## Batch runs

AutoFA analyzes completed evaluations; it does not launch the model evaluation.

### Setup and run

Follow the [quick start](../README.md#quick-start) to get the code and set your detector LLM endpoint
key. Use the batch command below in place of the interactive launch.

Requirements: [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and
Codex CLI installed and signed in. `uv run` installs Python 3.12 if needed and
sets up AutoFA's environment and dependencies automatically.

```bash
cd AutoFA
uv run --python 3.12 autofa run --config examples/run-configs/frontier-23859c40d27a4c84.yaml
```

Run `codex login` if needed. Before launching, replace the example's placeholder
`source.evalrdl_url` with your server/run URL, or use `source.local_folder` as below.

For Claude Code, sign in with `claude auth login` and set `config.harness` in
your copied YAML:

```yaml
config:
  harness: claude
```

This uses the configured Claude profile's model and reasoning defaults. Optional
`model` and `effort` fields can select other values supported by your CLI.

Readiness checks validate the harness login, advertised model capabilities, and requested
services before retrieving data. Custom deployment IDs remain allowed. Results are
saved under:

```text
~/.autofa/runs/<run-id>/
```

```bash
uv run --python 3.12 autofa dashboard --run <run-id>
```

Press `o` in the dashboard to open the failure cards in a browser.

### Run your evaluation

Copy the [example config](../examples/run-configs/frontier-23859c40d27a4c84.yaml);
its fields are strictly validated. Most keys under `config` use the same names as
`autofa.toml`; `optimization` selects a named objective preset for discovery.
Omitting `workflow` selects discovery; `workflow: classification` runs a configured
taxonomy selection, judge set and report.
Choose exactly one source:

```yaml
source:
  evalrdl_url: https://eval.example.org/#/run/<run-id>
```

or a local NeMo Gym/Harbor run:

```yaml
source:
  local_folder: /path/to/run
```

Relative local paths resolve beside the YAML and must contain exactly one recognized
run. Classification workflows may instead use `source.api_runs` with one or
more remote run IDs. Their complete schema is in
[Taxonomy classification](taxonomy-classification.md#run-the-complete-workflow-from-one-config).
See [`src/autofa/run_spec.py`](../src/autofa/run_spec.py) for the authoritative
field validation and source behavior.

Pin the discovery objective explicitly when using existing settings:

```yaml
config:
  optimization: recall
```

This selects 50% failure-task recall, 30% detector adherence and 20% normalized precision,
even if the loaded settings use the older Balanced preset. `balanced` and `precision` are
also supported. Omitting this field retains the loaded objective preferences under the
normal new-analysis policy. It does not modify global settings or objectives saved in other runs.

Keep credentials out of YAML. Configure required endpoint keys in the project's
`.env` or `~/.autofa/.env`. See [LLM endpoints](llm-endpoints.md) for other
providers. AutoFA stops before analysis if the detector endpoint is unavailable, because
`require_llm_detectors` defaults to `true`. `execution.require_llm_detectors: true` keeps that
requirement for one batch; set `require_llm_detectors = false` in `autofa.toml` to continue
with a warning instead.
Resume interrupted work with `uv run --python 3.12 autofa resume --run <run-id>`.

## How a session feels

For commands shown as `autofa` throughout these docs, use
`uv run --python 3.12 autofa` from the checkout.

```bash
uv run --python 3.12 autofa
```

Discovery investigates hypotheses within the existing per-round family and diagnostic
budgets. For a supported broad relation, the agent states observable evidence conditions,
implements a candidate, and checks inspected positives and contrasting negative controls.
If execution fails, it diagnoses missing evidence, implementation mistakes, and possible
classifier limitations before deciding whether to repair, defer, or reject the hypothesis.
An optional [classifier comparison](configuration.md#optional-classifier-comparisons-during-discovery)
uses configured profiles on fixed controls; it is not an extra mandatory step every round.
Normal population scoring and causal review still determine retained detector quality.

New `family_discovery(action="no_coherent_family")` calls require a structured
`decision`: the outcome, reason, next action, and relevant proposed relation, observable
conditions and cited evidence. A supported relation described only as too generic stays
unresolved and directs a bounded implementation test. Inspected incompatible causes may
justify rejection without a prototype; unavailable evidence and execution failures remain
explicit deferrals. The tool reports the required fields for each outcome. Historical
prose-only decisions remain readable and are not relabelled as tested conclusions.

Saved family decision records link checked evidence and actual diagnostic receipts.
Repeated unresolved conclusions point to a different concrete investigation without
granting fresh budgets. `smoke_test_feature` returns a `diagnostic_receipt_id` for explicit
author controls, along with the `source_sha256` needed when citing that execution.
Changes to the source evidence invalidate those citations.
These records make the agent's work inspectable by a supervisor while keeping candidate
development inside AutoFA. They do not turn agent-declared expectations into ground truth.

Pick an action from the home menu, finish it, and return to the menu until you Quit.

| Home action | What it does |
|-------------|--------------|
| **New analysis** | Choose analyses, then local NeMo Gym / Harbor logs or remote EvalRDL runs; start discovery when failure discovery is enabled |
| **Continue from library** | Seed from a local or visible cloud `library.json` and find more / improve |
| **Classify failures by taxonomy** | Assign failed rollouts to a fixed taxonomy and generate a report |
| **Explore failure library** | Open the local browser dashboard for saved libraries and reports |
| **Settings** | Configure harnesses, LLM endpoints, remote services, and analysis defaults |

CLI also supports `autofa run --config <run.yaml>`, `autofa library`,
`autofa publish --run <id>`,
`autofa continue --remote-library <id> [--remote-branch <id>] [--remote-version <id>]`,
`autofa resume --run <id>`, and
`autofa dashboard` outside the home menu.

**New analysis** and `autofa run --config` both produce the same per-run configuration.
After setup, they share harness readiness checks, source resolution, run creation, and execution.

Inside a session, type slash commands (Tab-completes) or plain English:

| Command | Purpose |
|---------|---------|
| `/discover` | Find top failure modes, then submit and score detectors *(agent)* |
| `/validate` | Re-evaluate and check claims with the selected behavior judge *(agent)* |
| `/compact` | Compact the current conversation while retaining its saved work |
| `/dashboard` | Interactive explorer (metrics, Mechanism prose, highlighted source) |
| `/diagnostics` | Show data-health diagnostics for the run |
| `/library` | List detectors and scores |
| `/clean` | Clear the working library (or `/clean <name>`) |
| `/save` | Save locally and publish the report to the configured remote library |
| `/goal` | Check Pareto progress vs previous round |
| `/help` | Help |
| `/exit` | End the session, save locally, and publish to the remote library (`/quit` also works) |

While discovery is running, **Ctrl+C** stops discovery and starts final detector scoring
on the full dataset, followed by final review. Another **Ctrl+C** cancels finalization and preserves resumable
checkpoints. Outside discovery, Ctrl+C cancels the current turn.

If a menu action fails, AutoFA keeps the error visible until you press Enter and
saves its traceback, including the original cause, to `~/.autofa/cli-errors.log`
(`$AUTOFA_HOME/cli-errors.log` for an isolated profile). This also covers dataset
preparation errors that occur before a run directory exists.

Temporary Data API failures (timeouts, broken connections, and HTTP
408/429/500/502/503/504) retry up to six total attempts with backoff. `Retry-After`
is respected up to a 60-second wait per retry. Interrupted exports restart in a
temporary file; a truncated download never replaces a completed cache entry.
Authentication errors, missing runs, and malformed JSON fail immediately.

If Claude loses a response mid-stream or reports a temporary service error,
AutoFA makes up to two recovery attempts in the same session. It preserves the
working library and round budget, and checkpoints the session ID as soon as it
is available so `autofa resume --run <id>` can recover after a process exit.
Cancellation and permanent errors do not retry. Exhausted retries and incomplete
turns remain visible as failures rather than successful analysis.

If Claude Code or Codex reaches its context limit, AutoFA asks the harness to
compact and retries in the same conversation. Saved libraries, helpers, repair
records, evaluation checkpoints, and saved discovery limits are retained. If compaction
fails or the request still exceeds the limit, the error remains visible; reduce
the request or evidence page size before retrying. The full
discovery protocol remains in the system prompt; `CLAUDE.md` or `AGENTS.md` contains only workspace
guidance and the live ledger to avoid injecting the protocol twice. Tool results
above 45,000 characters are saved intact and read through `read_tool_result` pages,
before Claude Code would move them into a file (above 50,000 characters). Their
envelope repeats short top-level fields such as `ok`, `advanced`, `round`, `status`
and counts exactly, plus the first 2,000 characters of `error`, `message` and
`next_action`; `truncated_fields` names the pointer for the rest.
Use `read_tool_result(result_id, describe=True)` for a structural index, then
`read_tool_result(result_id, pointer="/results/0/value")` to retrieve an exact field.
Follow `next_offset` with the same pointer and stop when it is null, even when
`complete` is false. Stop on errors or nonadvancing cursors; bound automated reads.
Join text pages before JSON decoding;
omit the pointer to retrieve the complete original response. Structure descriptions
contain paths, types and sizes, not evidence. `complete` describes the selected value
or child listing; `full_result_complete` identifies a complete original response.

Large repeated citations and smoke-test histogram strings use references to their
exact values. Follow `#/...` references from the original response root (drop `#`
when passing a pointer), even when reading a selected section. Compact review responses
also include `full_result`, an immutable copy of the original response, available after
detector edits or deletion. These presentation changes leave detector inputs and saved
reviews intact. Previews and summaries do not establish causal evidence or replace
required checks for later recovery, contradictions, or the complete trace.

For `sample_items` and `search_items`, an explicit `benchmark` narrows the selected
run corpus. A nonempty `benchmarks` list takes precedence; omitting both uses the
run's selected benchmarks. Filters outside that corpus return an error.

After sleep, a connection loss, Ctrl+C, or a process exit, reconnect and run:

```bash
autofa resume --run <run-id>
```

Resume restores the conversation and original unfinished request, working library,
round, and pending tool arguments. Use `autofa supervise finalize --run <run-id>`
or `autofa publish --run <run-id> --saved-only --json`
to finish or deliver saved findings without choosing further exploration.
A second session cannot open the same run
while another session holds it. Sleep still pauses local work;
AutoFA does not keep a sleeping laptop running.

Successful LLM detector evaluations are committed per item and reused when evaluation
is retried. Completed judge stages and frozen claim compilations are checkpointed too:
an interrupted confirmation retries that independent stage, keeping the completed
initial review. Reuse requires matching source, dependencies, evidence, model and
endpoint, key variable name, and request options for detector evaluations;
judges additionally match claim, guidance, rubric,
and evidence-policy versions and revalidate saved citations. Changed or damaged
checkpoints are recomputed. Processing errors are retried; a valid uncertain judgment
is retained as uncertain. Judging progress appears in `status.md` and each batch's
`behavior_judge/<batch>/progress.json`.

Within a run, MCP evaluation, review and mutation operations reserve their detector
for the whole operation, including automatic evaluation and judging during submission.
A conflicting request returns `operation_in_progress` before starting expensive work;
use the original operation's result instead of repeatedly submitting the same request.
Unrelated detectors can still run concurrently. Library-wide changes and shared
dependencies coordinate with these reservations. Cancellation releases a reservation
after the worker has stopped; completed evaluation and judge checkpoints remain reusable.

Interrupted repair validation keeps its reserved examples, their order, and the
original exclusion history. Resume completes that same sample without consuming a
new repair attempt. Missing reserved examples or changed candidates/policies must
be resolved before validation can continue.

Run configuration, libraries, score history, session IDs and recovery checkpoints use
atomic file replacement. Completed source downloads and imported trace chunks remain
reusable; an interrupted individual download may restart. An interrupted model request
without a completed checkpoint may also need repeating. Resume reconstructs evaluation
aggregates from saved successful items rather than publishing a partial evaluation.

To debug a specific EvalRDL run ID without returning to the home menu:

```bash
set -o pipefail
autofa new --api-run <remote-run-id> --analysis-focus both --once \
  2>&1 | tee /tmp/autofa-debug.log
```

`/save` saves the run and catalog locally, then publishes the current report to
the configured remote library (AutoFA Cloud or a self-hosted AutoFA server) without ending
the session. Discovery completion or a graceful discovery stop first completes any missing
full-corpus detector scans. This includes capped Python/regex
evaluations and sampled LLM evaluations; current complete scans and detector checkpoints
are reused. Finalization then checks missing or unfinished detector reviews, preserving
completed judgments and recovering unfinished cases within a bounded review budget.
Both stages show progress. `/exit` or `/quit` then publishes the report unless it is
unchanged since the last successful publication. A hard interruption preserves the
checkpoint without requiring final scoring to finish.
Saving and publishing themselves use existing evaluation results: `/save` and
`autofa publish` do not run detectors, start judge reviews, or retry failed reviews.
Final review does not change detector definitions or reroll completed semantic judgments.
If required evidence is unavailable or review still cannot finish, AutoFA saves the results
and shows the specific missing-score reason in the terminal and HTML report rather than
claiming every score is complete. Repeating finalization reuses its saved results.
Missing credentials or upload failures are reported while the local run and
catalog remain safe; retry with `/save` or `autofa publish --run <id>`.
Browse-library **`p`** also publishes immediately.

## Discovery limits and continued campaigns

Discovery defaults to a minimum of **1 Good failure detector**. Settings saves your
preferred minimum, or use `autofa new --min-failure-modes 5` (also supported by
Continue and Resume). `0` disables this minimum. For a target of 5 Good detectors with
3 planned rounds and only 1 supported Good detector, normal discovery starts another turn and keeps
exploring through the bounded recovery allowance when the requested minimum is still unmet.
Saved supervisor pause/stop/finalize requests remain authoritative.

**Continue until coverage plateaus** in Settings defaults **Off**. Regular discovery ends
at the planned round limit once the minimum Good count is met, then performs final full-dataset
scoring. It does not wait for unrelated pending reviews or synthesis. Turn it **On** to keep
searching until coverage progress and pending-work requirements are also satisfied.
The choice is saved per run for Resume;
Continue from a library uses your newly selected settings. Running processes need restarting
to load this code; legacy runs without the setting use Off when resumed.

In normal interactive discovery, two consecutive turns without changes to the library,
round, evaluations, failure-mode ledger, or repair checkpoints pause automatic follow-ups
and return control to the prompt. The library is saved and the target remains unmet;
use `/save` to publish what is available, or give new guidance and retry `/discover`.
Rejected candidates and ongoing repairs still count as work, even before a new mode qualifies.

If Claude reports a login failure during discovery, sign in from another terminal with
`claude auth login` (use the configured binary shown in the error), then retry `/discover`
or resume the saved run. The conversation and saved detectors are retained. AutoFA's
prompt does not implement Claude's `/login`; `autofa login` authenticates the library
service. API-key users should correct their key and restart AutoFA.

The minimum uses the same **Good** thresholds as the CLI and report. Each detector must
also meet the evaluation and review evidence requirements below to count:

| Criterion | Required evidence |
|---|---|
| Coverage | Strictly >1%, using known matched failures divided by **all failures in the selected dataset** |
| Normalized precision | `ppv_at_pi0` ≥70% |
| Detector adherence | ≥90% in a current, completed claim review |
| Cross-task support | Matched failures from at least **five distinct source tasks**; adherence is reviewed separately |

Adherence review has no minimum Coverage or distinct-task count. Retained, scored OK
detectors remain visible in reports without counting toward the Good minimum.
Automatic submission reviews require normalized precision ≥70%, current scoring and usable
matches, subject to the existing prefilter and review prerequisites. Explainers use their
existing within-parent normalized precision for scheduling. Coverage and task support
still determine grades and the Good minimum above.
The response exposes `review_schedule` and `review_deferred`
with a repair for the actual limiting dimension, supported specialization, or evidence
investigation to pursue. `grade_gap.target_grade` identifies the runtime's current aspiration;
`next_grade` remains the immediate milestone. Missing Coverage, precision, adherence or task
support directs the next action; it does not default to precision work. For a coverage gap,
broaden the supported mechanism or synthesize related seeds into a shared-mechanism parent,
comparing the same cohort and deduplicating tasks. An OR of unrelated detectors is not a
shared mechanism. A fair comparison on the original panel does not validate added scope:
inspect new matches across distinct tasks, including passing and boundary counterexamples.
The broadened claim needs its own review through existing paths; seed confirmations do not
transfer. Selected diagnostics do not estimate population performance. Use existing capped
evaluation and review during discovery, then the normal final full-corpus validation.
A coherent subtype may improve precision at lower coverage if its claim is revised consistently.
For precision from 40% to below 70%, deliberately request a diagnostic review when per-case
claim evidence will guide a repair, or standard review when validating a useful OK detector.
Explicit `judge_function(name)` still defaults to standard review. Deferral does not reject
or delete the candidate; existing hard prefilters, grades, judge policy, frozen review recovery,
and final scoring/review remain unchanged. See [review scheduling](metrics.md) for details.

Final qualification requires a current evaluation attempted over the whole selected dataset.
Current sampled estimates can establish provisional grades during discovery. Execution errors
or unvisited rows do not establish final scores; honest abstentions may remain.
Repeated rollouts of one task count once. Previously inspected tasks can provide
causal support; independent confirmation on unseen tasks is a separate property. Abstentions
keep the full failure denominator and do not erase verified positive causes.
Small review samples remain visible. Useful one-task
findings stay available as diagnostics and synthesis seeds, but do not fill the target.
Unjudged, stale, unfinished, rejected, structural-proxy, and exact duplicate candidates
also do not qualify. `/goal` shows the count and shortfall. The minimum is saved per run
and preserved on Resume; an explicit
`autofa resume --run RUN_ID --min-failure-modes 2` updates that run's target.

For longer campaigns, use the [AutoFA driver](../agent-skills/autofa-driver/SKILL.md).
Give it a coverage goal, round allowance, deadline and spending constraints. It drives native
New/Continue/Resume workflows, monitors progress and uses saved lifecycle controls to pause,
stop or finalize. Unattended monitoring requires a persistent host execution facility.

Configure `discovery_plateau_patience` for completed investigations
without additional confirmed task coverage (default `6`).

`execution_history.jsonl` records the Python executable, installed source directory,
working directory, and available Git revision for each session launch, including Resume.
To select this checkout explicitly, launch `uv run --python 3.12 autofa` from its root.

## Efficiency metrics

AutoFA can generate efficiency metrics without running failure discovery or
add them as a separate tab in the failure-card report.

Efficiency metrics require one canonical EvalRDL run, either imported from local
NeMo Gym / Harbor logs or downloaded from a remote server. For multiple runs or
standalone Parquet/JSONL inputs, select failure discovery only.

**Efficiency metrics** is Off by default and **Failure discovery** is On.
Before dataset selection, open **Analysis settings > Analyses to run** to toggle
either for this run. Use **Settings > Analyses to run** to save defaults.
CLI flags override these defaults:

```bash
# Efficiency report only
autofa new --analysis-output efficiency

# Failure discovery only
autofa new --analysis-output agent

# Failure discovery with a separate Efficiency metrics report tab
autofa new --analysis-output agent --analysis-output efficiency
```

The report compares passing and failing rollouts across available cost, token,
agent-activity, and context metrics. It includes raw differences in means and
distribution comparisons. Runs with no recorded tool calls omit the agent-activity
section and final context length.

Cost uses configured model rates when available and clearly labeled stand-in rates
otherwise. Each report also provides expandable data coverage
and provenance, and each section lists its metric rules and exact values; unavailable
metrics are omitted instead of invented.

See [Efficiency metrics](efficiency.md) for report locations, metric
definitions, pricing configuration, provenance rules, and interpretation guidance.

## Taxonomy classification

`autofa classify` labels failed rollouts with a fixed failure taxonomy using an
LLM judge, then writes an offline HTML report. New run inputs classify 10 random
failures per run by default; `--per-run N` changes the random bound, `--seed N`
makes the sample repeatable, and `--all` selects every failure. Saved selections
are reused as-is unless explicit refinements are passed. Repeated judge or
taxonomy profiles use separate `--output` files. Sharded runs require a
shard-specific `--output`.

```bash
autofa classify --api-run RUN_ID
autofa classify report taxonomy_classifications.jsonl
autofa run --config taxonomy-classification.yaml

# After reviewing the report, reuse that config's result files as a discovery bucket:
autofa new --config taxonomy-classification.yaml --label OWN-X
autofa continue --config taxonomy-classification.yaml --group accepted --evidence complete
```

The run-config form supports multiple independent judges and optional report
publication in one command. New/Continue reuse its configured judge output files;
choose the judge, group, labels, and evidence filters after reviewing the report.
The selected failures become a child cohort and receive same-run successful controls
before failure discovery starts. See
[docs/taxonomy-classification.md](taxonomy-classification.md).

## Preflight checks

In **New analysis** or **Continue from library**, open **Analysis settings > Analyses to run** to configure
**Efficiency metrics** (Off), **Failure discovery** (On), and
**Preflight > Cheating** (Off). Use **Settings > Analyses to run** to save defaults.
Enabled preflight checks run before the other workflows and save reports with the run.
Each check also has a standalone command under `autofa preflight`.

`autofa preflight cheating` finds rollouts where the agent gamed the evaluation (looked up
the fix, edited the tests, ...). Patterns screen every rollout, an LLM judge
reviews the flagged ones, and an HTML report lists the cheated rollouts.

```bash
autofa preflight cheating --api-run RUN_ID
autofa preflight cheating --input ./eval-results --max-reviews 50
```

See [docs/cheating-check.md](cheating-check.md).

## Who does what

| Role | Backend |
|------|---------|
| Discovery agent (`/discover`, chat) | Selected Claude Code or Codex harness |
| Behavior judge (`judge_function`) | Selected judge harness; follows discovery by default (fresh sessions) |
| LLM detectors / `llm_classify` | Configured OpenAI-compatible endpoint (`feature_*`) |
| Improvement classifier | Configured OpenAI-compatible endpoint (`judge_*`) |

## Tips

- Start with **New analysis** → **Local eval logs (NeMo Gym / Harbor)** or **EvalRDL remote server** → select runs → `/discover`
- For SWE: type `benchmark:swebench-verified` in the remote picker, or `autofa new --api-run RUN_ID`
- Read submit feedback (`verdict` + metrics + novelty warnings); drop **EMPTY / WEAK** detectors
- Open `/dashboard` to read **Mechanism**, **How the detector works**, bucket precision, and source
- Use `/goal` when iterating rounds; stop when the frontier stops improving
- To disable creative SSoT exploration: `exploration_ssot = false` in `autofa.toml`
- For scripts that should not return to the menu: `autofa library --once`

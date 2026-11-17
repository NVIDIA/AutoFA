# Runtime estimates

AutoFA estimates runtime. Discovery can choose new detectors and make
additional judge calls during a run, so partial API charges do not establish the
cost of the analysis.

The depth picker displays a median and the **observed range**, only when at least
three runs match the selected discovery/judge models, reasoning settings,
worker counts, evaluation limits, minimum failure-mode target and detectors per
round. Each comparison must have measured all the requested rounds. This measured
forecast does not extrapolate round counts or substitute another model.
The observed range is descriptive, not a statistical confidence interval or a
promise that the next run will fall inside it.

Workload matching requires row count, sampled trace length and sampled step count
to each be within a factor of 1.25 of the selected workload. This is an eligibility
gate, not a runtime scaling formula. In particular, similar row counts cannot
make a deep agent trace equivalent to a short answer. Profiling counts only the
selected model/benchmarks and samples up to 48 traces, allocating samples
proportionally across files and Parquet row groups. Within a selected row group,
it reads a bounded prefix; these samples are not a full trace census.

When there are fewer than three comparable runs, the picker shows a **rough**
planning range. It prefers the same settings and then the same discovery model
among similar workloads. Otherwise it uses similar logs, or the full reference
pool if no workload matches. The UI identifies uncalibrated model/settings or
workload combinations. It projects the fastest/slowest measured round durations
across the requested number of rounds, assuming similar pace per round. Bounds
round outward to five minutes for totals under an hour and quarter-hours internally.
The UI presents hour-long estimates as whole-hour ranges, rounding bounds outward
and omitting minute suffixes. The timing guide above the choices explains model,
log-size/depth, validation and extra-round uncertainty. A pricing disclaimer
explains that model rates, token usage and extra rounds affect cost; no numeric
cost forecast is presented.
This is a starting planning range, not an observed total or a calibrated model
speed. It does not invent model speed factors or log-size multipliers. Without
any recorded round timings at all, no numeric estimate is shown.

During discovery, both Claude and Codex show the current round (for example,
**Round 1 of 4**) and an **ETA** expressed in minutes remaining, with an approximate
local finish time and a separate uncertainty range. It recalculates every second;
cached input files are reread only when they change. Minute-level presentation is
display granularity, not a claim of minute-level forecast accuracy. The first round
starts with the historical estimate, subtracting measured active time. After two
completed rounds, the central estimate uses the median of the latest three completed
rounds (or both available rounds), while the range retains this run's observed
fastest/slowest pace. This lets the central estimate follow changing pace without
hiding variability. A resume forecasts only the rounds remaining when its timing
context began. If a round exceeds the observed range,
it reports the overrun instead of sitting at zero minutes remaining. Additional
rounds to reach the saved minimum failure-mode target have no reliable ETA.

## What is actually timed

New runs save settings and workload metadata in `estimate_context.json`. Timing
uses the existing `status_log.jsonl`: `started`/`finished` harness events delimit
active work and `advance_round` events delimit completed rounds. Time between
turns is excluded. An agent turn finishing alone does **not** prove the current
round finished. The last round is omitted unless its boundary was recorded.
Failed/interrupted rounds, overlapping turns, missing or out-of-order boundaries,
and resumes with different recorded settings do not supply calibration data.

A context starts at its recorded log offset; timings before that offset cannot
be attributed to the context's settings.
Runtime estimation does not collect a token or cost ledger.

## Calibration baseline

The bundled baseline contains two usable timing sequences with recorded round
boundaries:

| Discovery model | Closed rounds, active minutes | Selected rows | Sampled steps/trace |
| --- | --- | --- | --- |
| Claude Opus 5.5 | 58.0, 42.4 | 3,000 | 36 |
| Claude Opus 5.5 | 124.4, 101.8, 61.1 | 2,400 | 184 |

The third example ran for 205 minutes but recorded no round boundary, so it is
not a completed-round sample. These examples also lack some saved reasoning
and judge settings. Their anonymous measurements remain in the bundled audit
baseline and can supply the explicitly rough fallback. They cannot match the
complete signature required for the measured forecast. For an unmatched workload,
the bundled reference pool displays roughly 30 minutes–3h for one round or
2h–7h for three rounds; live measurements replace this broad assumption.

The bundled audit baseline (`src/autofa/estimate_baselines.json`) was derived from
explicitly supplied run directories and Claude transcript storage, using the actual
model identity, never today's defaults. It contains only aggregate workload sizes,
model settings and measured durations; no paths, prompts or dataset identities.

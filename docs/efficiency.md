# Efficiency metrics

The efficiency metrics workflow compares how correct and incorrect rollouts use tokens,
context, turns, tools, and cost. It can run by itself or alongside failure discovery.
Select exactly one evaluation run; efficiency metrics do not combine runs.

Efficiency metrics accept only canonical EvalRDL inputs. AutoFA automatically
imports supported local runs into EvalRDL and retrieves trace batches for remote runs.
Standalone Parquet files and unrecognized logs remain available to failure discovery,
but cannot be used for efficiency metrics.

AutoFA checks these requirements, and that the run has rollouts, before downloading
data or starting an analysis. When the data cannot produce efficiency metrics, an
efficiency-only run stops and shows the reason. A run that also includes failure
discovery asks whether to continue with failure discovery only; batch and noninteractive
runs stop instead.

After the data is prepared, AutoFA shows a one-line summary before the analysis starts:
rollouts, correct and incorrect counts, and how many rollouts recorded token usage. It
warns when token usage is missing or incomplete, or when either group has fewer than 20
rollouts, which is too few for the report summary. When no reward is recorded, or every
rollout has the same outcome, there is nothing to compare, so AutoFA asks whether to
continue or stop; a run that also includes failure discovery can continue with failure
discovery only. Batch and noninteractive runs continue and print the warning.

For [setup](../README.md#quick-start), use the README quick start. Prefix the CLI
commands below with `uv run --python 3.12` when running from the checkout.

## Choose what to run

When you start **New analysis** or **Continue from library**, the initial
**Analysis settings** screen contains an **Analyses to run** submenu. Open it before
selecting data. **Efficiency metrics** is **Off** by default and **Failure discovery**
is **On**; press Enter to toggle either:

- **Efficiency metrics** creates an efficiency report.
- **Failure discovery** finds failure modes, writes detectors and validates them.
- Selecting both produces one report with separate **Failure modes** and
  **Efficiency metrics** tabs.

Within **Analyses to run**, **Preflight** contains optional preflight checks, initially **Cheating** (Off).
Keep at least one analysis or preflight check On. Enabled preflight checks run
on the selected data before the other workflows; see [Preflight: cheating](cheating-check.md).

Choose **Back**, then **Continue** to apply your choices to this run. To save defaults
for future New/Continue runs, open **Settings > Analyses to run**. There is no separate
analysis-selection prompt after the dataset is selected. Without explicit CLI flags,
noninteractive New/Continue also uses the saved defaults (initially failure discovery only).

The equivalent command-line forms are:

```bash
# Select the source interactively, then generate efficiency metrics only.
autofa new --analysis-output efficiency

# Failure discovery only, including inputs that do not support efficiency metrics.
autofa new --analysis-output agent

# Run failure discovery and add efficiency metrics to its report.
autofa new --analysis-output agent --analysis-output efficiency
```

## Reports and saved data

Efficiency-only analysis opens `efficiency.html`. Combined analysis adds an
**Efficiency metrics** tab to `failure_cards.html`. AutoFA also saves
`efficiency.json` for programmatic use.

The terminal prints the report link when it is generated. After an efficiency-only
run, choose **Publish** or **Return to home**. In a combined run, `/save`, `/exit`,
or `/quit` publishes the report; `/exit` and `/quit` skip publication when nothing
changed since the last one. A completed run can be published again with
`autofa publish --run <run-id>`.

## What the report shows

When the selected evaluation recorded the required data, the report shows:

- overall accuracy and best-of-N;
- **Cost** — token cost using the applicable model prices;
- **Token usage** — input, cached input, uncached input, reasoning, answer, output, and
  total tokens. Uncached input is input minus cached input. The cached and uncached input
  charts are omitted when cached tokens were not recorded.
- **Agent activity** — assistant turns, tool calls, calls per turn, tool-argument and
  tool-output characters, and calls by tool type.
- **Context** — prompt size and, for agentic runs, final context length.

Each metric has a card with one point per rollout, the distribution for correct and
incorrect rollouts, and three values:

```text
mean correct · mean incorrect · difference = mean incorrect − mean correct
```

A positive difference means incorrect rollouts used more. A **p < 0.05** badge marks a
statistically significant distribution difference. Hover over a chart for the median,
middle 50%, and individual values. Treat comparisons as investigation leads, not proof
that a metric caused a failure.

Each section ends with **Metric rules & exact values**, which defines its metrics and
lists totals and correct and incorrect medians.

The **Summary** box, after **Data coverage & provenance**, applies the same rules to every
run:

- A metric needs 20 correct and 20 incorrect rollouts with a recorded value; otherwise it
  is listed under **Too few rollouts**. Metrics with p ≥ 0.05 are listed under
  **No significant difference**.
- A metric recorded for only some rollouts is judged on those rollouts, and the summary
  shows the share it is based on, for example "input tokens (↓, 71% of rollouts)".
- A metric differs significantly when p < 0.05, the distributions differ by at least
  0.1 (the test statistic), and the medians differ by at least 5%. Significant metrics
  below either size threshold are listed as **negligible**.
- Metrics that differ are grouped by the test statistic: small from 0.1, moderate from 0.2,
  and large from 0.4. Within a group, the largest comes first. ↑ means incorrect rollouts
  have the higher median; ↓ means lower.
- A section is named instead of its metrics only when all of them differ in the same
  direction with the same size.
- Cost is not listed with the other metrics. Its line compares the share of rollouts that
  are incorrect with their share of total cost.

### Efficiency on failure cards

In a combined report, each failure card has a collapsible **Efficiency** section after its
detector. With at least 20 matched failed rollouts, it charts total tokens, assistant turns,
tool calls, and cost for three groups: the run's correct rollouts, the rollouts matched by
this failure mode, and the run's other failed rollouts. The **p < 0.05** badge compares the
matched rollouts with the correct ones. With fewer matched rollouts, the section says that
efficiency cannot be computed. When the detector was evaluated on a sample, the section uses
the sampled matches and says so.

### Non-agentic runs

AutoFA treats a run as agentic only when at least one tool call was recorded. If no
tool calls were recorded, it omits the **Agent activity** section and final context
length.

## Cost and model prices

AutoFA uses the model's entry in `efficiency_model_prices` when one is configured.
Otherwise it uses clearly labeled stand-in prices: $0.15 input, $0.03 cached input,
and $0.50 output per million tokens.

The report labels the stand-in case **Simulated cost**. It is useful for comparisons,
but it is not a provider bill.

Configure real rates in the active `autofa.toml` by replacing the default
`efficiency_model_prices = {}` line with:

```toml
[efficiency_model_prices."provider/model-id"]
input = 1.0
cached = 0.1
output = 3.0
```

## Data provenance and missing values

When a metric is recorded for only some rollouts, an **Incomplete data** notice at the
top of the report lists it with how many correct and incorrect rollouts it covers.

Open **Data coverage & provenance** to see which values were recorded, derived,
estimated, unavailable, or not applicable, together with their coverage and
fallback. Each section's **Metric rules & exact values** gives the exact calculation
rules.

AutoFA keeps recorded zeros, omits missing values, and only derives or estimates
values using rules disclosed in the report. It does not silently invent unavailable
metrics.

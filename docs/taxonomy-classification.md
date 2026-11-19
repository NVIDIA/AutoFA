# Taxonomy classification

`autofa classify` gives each failed rollout one failure mode from a fixed
taxonomy. An LLM judge picks the mode, and AutoFA writes the results plus an
HTML report you can open offline.

After [setup](../README.md#quick-start), prefix the commands below with
`uv run --python 3.12` when running from the checkout.

```bash
autofa classify                                  # pick runs and a count interactively
autofa classify --input ./eval-results           # local NeMo-Gym / Harbor run
autofa classify --api-run RUN_ID --all           # every failure of a remote run
autofa classify --input SELECTION_ID             # reuse a saved selection
autofa classify report results.jsonl more/       # rebuild the report
```

The home menu entry **Classify failures by taxonomy** runs the same command. It
does not start an agentic `/classify` session.

## Choosing rollouts

By default AutoFA classifies healthy, scored rollouts with `reward <= 0`. To use
another threshold, pass one or more `--reward` filters, for example
`--reward '<1'` or `--reward '>=0' --reward '<0.5'`. AutoFA warns you when a
benchmark has partial rewards and you did not set a threshold.

Runs and rollouts use the same filters as `autofa prepare` (`--api-run`,
`--input`, `--model`, `--benchmark`, `--task`, `--latest`, and so on).

How many rollouts:

- Interactive mode asks you.
- A new run classifies 10 random failures per run. Use `--per-run N` to change
  the number or `--all` for every failure.
- A saved selection is reused as is. `--per-run N` then takes its first N rows,
  or a random N with `--sample random`.
- `--seed N` makes random sampling repeatable. On a saved selection it also
  needs `--sample random`.
- `--dry-run` shows the counts without downloading anything or calling a judge.

## What the judge sees

The judge reads the rollout trace (messages, tool calls, results) and whatever
the grader recorded:

| Source | Trace | Grader evidence |
| --- | --- | --- |
| Remote (`--api-run`) | Full prepared trace | Rollout artifacts, verifier text, raw record |
| Local (`--input`) | Imported trace | Imported artifacts, raw record from the original folder |

Automatic artifact-text hydration is restricted to verifier artifacts and the
exact output artifact `model.patch`. Golden and reference patches are never
added to judge evidence.

**Long traces** are cut into overlapping windows that are judged in order.
`--window-max-chars N` sets the window size: 512,000 characters by
default, which is usually about 100K to 150K input tokens including the prompt,
and up to about 270K on dense code traces. Use a
smaller value, such as 32,000, for judges with small context windows. A single
event that is too large for a window is split into chunks, so its middle is not
lost.

The primary `ordered_trace` keeps its existing chronology. Non-evaluation
auxiliary lanes are presented as separately ordered, lane-qualified sections;
attempts, relations, provenance, tool definitions, and retained prompt
resources are included as bounded standardized sections. HTTP and agentic
workflows receive the same safe projection. Agentic judges additionally get
exact read-and-cite tools for those events and resources.

**Missing evidence** is recorded, never guessed. Each record has an
`evidence_receipt` that lists what the judge received (`trace`, `verifier`,
`artifacts`, `raw_outcome`), each marked `included` or `absent`. If the grader
gave only a reward and no reason, the record gets the warning
`outcome_cause_unavailable` and the judge may abstain. If there is no usable
trace or task, the judge is skipped and the record is marked `needs_evidence`.

## Judge setup

Configure judge models and endpoints under **Settings > LLM endpoints**. Keys
come from the environment or AutoFA's `.env` file.

- AutoFA's taxonomy is based on Scale AI's
  [*Model or Harness? An Interaction-Centric Taxonomy for Localizing Agent
  Failures*](https://arxiv.org/abs/2607.28802). `taxonomy_v0.json` preserves
  the paper's Appendix B taxonomy. The default `taxonomy_v1.json` refines it
  using findings from AutoFA classification reports and manual analysis.
- `--taxonomy FILE` replaces the default v1 taxonomy with a custom taxonomy.
  To propose a change to the built-in taxonomy instead, see
  [Proposing a failure mode](#proposing-a-failure-mode).
- `--judge-model MODEL` overrides the model for every active LLM pass. Without
  it, LLM passes use `judge_model`. Agentic passes always keep the separate
  Judge endpoint model.
- `--judge-param KEY=VALUE` adds a parameter to HTTP judge requests. Repeat the
  flag to add more than one. For example:
  `--judge-param reasoning_effort=high --judge-param max_tokens=32000`.
  Use `--judge-param api=responses` for the Responses API. Put
  provider-specific parameters in `extra_body`, for example
  `--judge-param 'extra_body={top_k=20}'`.
- `--timeout` controls how long AutoFA waits for each HTTP judge call
  (default: 300 seconds). Failed calls are retried three times.
- `--workers` controls the number of parallel judge calls (default: 4).

AutoFA checks the judge with one small call before downloading anything.

Agentic passes use the **Judge endpoint** harness (Claude Code or Codex)
instead of the HTTP judge.

## Pass setups

A pass setup is a first pass, optionally followed by a second pass that reviews
it. The default is LLM first pass plus LLM second pass.

| Pass setup | Command |
| --- | --- |
| LLM only | `--first-pass llm --second-pass off` |
| LLM + LLM review (default) | `--first-pass llm --second-pass llm` |
| LLM + agentic review | `--first-pass llm --second-pass agentic` |
| Agentic only | `--first-pass agentic --second-pass off` |
| Agentic + LLM review | `--first-pass agentic --second-pass llm` |
| Agentic + agentic review | `--first-pass agentic --second-pass agentic` |

The defaults come from `taxonomy_first_pass` and `taxonomy_second_pass` in
`autofa.toml` (`llm` + `llm`). The interactive flow exposes the same six
combinations. Agentic passes require the configured Judge endpoint harness.

**Which one to use.** LLM + LLM is the budget-friendly default, including for
multimodal benchmarks. Agentic first pass can improve accuracy on coding and
other tool-heavy traces; add an LLM or agentic review when the expected gain
justifies the extra cost. These are recommendations, not restrictions.

- **LLM** sends the evidence to the HTTP judge.
- **Agentic** starts a restricted harness session that can search the trace and
  the retained artifacts. It can read lane-qualified event parts and
  standardized resource fields, follow tool calls to their results, check cited
  ranges, and open or render artifacts. It has no general
  filesystem access and cannot change anything. As a reviewer, it starts from the
  first-pass answer, checks the cited events, and searches wider only when
  those events neither confirm nor contradict the answer.
  `--agentic-max-turns N` limits Claude turns or Codex evidence-tool calls per
  pass (32 by default). Both harnesses also have a 10-minute session timeout.

A second pass can confirm, change, or withdraw the first-pass answer. Each
rollout still ends with exactly one failure mode. On long traces, the reviewer
can look at later windows to see whether an earlier failure was recovered. If
the review fails, the first-pass answer is kept and the error is recorded under
`second_pass`.

Agentic results are well-supported hypotheses, not gold labels.

When trusted visual artifacts are available, the HTTP judge receives them
automatically. If the endpoint rejects image input, AutoFA warns once and
continues with text evidence while marking the visuals as not shown. Agentic
judges inspect whatever retained artifact tools are available for that rollout.

### Run the complete workflow from one config

`autofa run --config taxonomy-classification.yaml` selects and caches rollouts, runs
every configured judge, and builds each report:

```yaml
version: 2
workflow: classification

source:
  api_runs: [RUN_A, RUN_B]

selection:
  per_run: 10
  sample: random
  seed: 42
  cache: classification-results/evidence

taxonomy:
  # Optional HTTP-judge overrides; omitted values use autofa.toml.
  judge_base_url: https://judge.example/v1
  judge_api_key_env: JUDGE_API_KEY
  runs:
    - first_pass: llm
      second_pass: agentic
      judges:
        - name: default-judges
          # Optional override for every active LLM pass.
          # model: openai/gpt-5
          # Optional agentic harness override.
          harness: codex
          # Arbitrary KEY=VALUE settings for every LLM pass.
          params:
            - api=responses
            - reasoning_effort=high
            - max_output_tokens=32000
          workers: 4
          output: classification-results/default.jsonl
      report:
        output: classification-results/default.html
        # publish: reviewed-taxonomy
```

Replace `source.api_runs` with exactly one `source.evalrdl_url` or
`source.local_folder` to use an EvalRDL run URL or local NeMo Gym/Harbor
results. Relative cache, output, report, and local source paths resolve beside
the YAML. Each run accepts all six LLM/agentic pass combinations. Add judge
entries to compare models over the same cached cohort; each judge may override
`model`, `harness`, `params`, `workers`, and `output`.

### Reviewing an existing first pass

You can add a second pass to saved first-pass results without redoing the first
pass:

Because the second pass sees the first-pass answer, both passes form one
reviewed classification rather than two independent votes in a multi-judge consensus.

```bash
autofa classify --input first-pass.jsonl --second-pass llm
autofa classify --input first-pass.jsonl --second-pass agentic
```

If the first pass used an evidence cache, point to the same cache so the
review sees identical evidence. This also works on another machine, without the
original selection, as long as the cache folder comes along:

```bash
autofa classify --input first-pass.jsonl --second-pass agentic \
  --cache classification-results/evidence
```

The review is written next to the input, for example
`first-pass.second-pass-agentic.jsonl`, and each record stores the hash of the
first-pass record it reviewed. If the input mixes several judges, choose one
with `--judge-view`.

## Output files

Results go to `taxonomy_classifications.jsonl` in the selection folder
(`$AUTOFA_HOME/cohorts/<id>/`), or to `--output`. Next to it, AutoFA writes:

- `taxonomy_classifications.manifest.json`: which runs and rollouts were used,
  the taxonomy, the judge settings, and a hash of the results.
- `taxonomy_classifications.html`: the report, rebuilt after every run. It is a
  single file that works offline.

Keep one JSONL file per taxonomy and judge setup. Use a different `--output`
for each, as in the example below.

## Re-running, sharding, and caching

- `--resume` skips rollouts already classified with the same taxonomy and
  judge. `--resume-from PATH` also reuses matching results from another file or
  folder.
- `--limit N` classifies the next N rollouts still pending, so repeated resumed
  runs work through a large selection in batches.
- If the last JSONL line was cut off, `--resume` warns and removes it. A broken
  line earlier in the file stops the run.
- Sharded runs (`--shard-count`, `--shard-index`) need their own `--output` per
  shard. Each shard gets its own JSONL, manifest, and report.
- `--cache DIR` classifies from the evidence already prepared in DIR
  and prepares and saves whatever is missing, so several judges and passes see
  the same evidence. Entries that are stale, incomplete, or built with another
  window size are prepared again. `--prefetch-only` fills the cache without
  calling a judge.

## Example: compare several judges

This picks up to four small failures from two runs, prepares the evidence once,
and classifies the same rollouts with three judges. Set up the endpoint as in
[Judge setup](#judge-setup) and swap in models your endpoint serves.

```bash
autofa classify \
  --api-run RUN_A --api-run RUN_B \
  --where 'completion_tokens <= 10000 AND n_truncated_events = 0' \
  --per-run 2 --sample random --seed 42 \
  --cache classification-results/evidence \
  --prefetch-only
```

The command prints a selection ID. Use the same ID for every judge so all
results describe the same rollouts:

```bash
autofa classify --input SELECTION_ID \
  --cache classification-results/evidence \
  --judge-model us/azure/openai/eccn-gpt-5.6-luna \
  --output classification-results/luna.jsonl

autofa classify --input SELECTION_ID \
  --cache classification-results/evidence \
  --judge-model azure/openai/eccn-gpt-6.1-sol \
  --output classification-results/sol.jsonl

autofa classify --input SELECTION_ID \
  --cache classification-results/evidence \
  --judge-model nvidia/deepseek-ai/eccn-deepseek-v4-flash \
  --output classification-results/deepseek.jsonl
```

Each judge writes its own JSONL under the same results folder, along with its
manifest and report. Point `report` to that folder to discover all result
JSONLs, or list only the JSONLs you want to combine:

```bash
autofa classify report classification-results \
  --output classification-results/combined.html

autofa classify report \
  classification-results/luna.jsonl \
  classification-results/sol.jsonl \
  --output classification-results/luna-sol.html
```

Add `--publish NAME` to upload the report to the AutoFA library, using the same
login as `autofa publish`. Publishing the same NAME again adds a new version.

The `--where` filter keeps very large or already truncated traces out of a
small comparison. Loosen it for bigger campaigns.

## Continue with failure discovery

After reviewing a classification report, reuse the same classification config and
choose a bucket at invocation time:

```bash
autofa new --config taxonomy-classification.yaml --label OWN-X
autofa continue --config taxonomy-classification.yaml \
  --group accepted --evidence complete
```

New/Continue read the judge output paths from the config. The usual
`--judge-view`, `--group`, `--label`, and `--evidence` filters choose the
bucket. AutoFA creates a child cohort, adds same-run successful controls, and
then starts failure discovery.

## Proposing a failure mode

To suggest a new failure mode, or a split, merge or redefinition of an existing one, in the
built-in taxonomy, open a
[Failure-mode proposal](https://github.com/NVIDIA/AutoFA/issues/new?template=failure_mode_proposal.yml)
issue. Maintainers agree to the proposal there before anyone opens a pull request that edits
`taxonomy_v1.json`. A strong proposal:

- names the interaction (for example *tool and model*) and the component at fault, following
  the taxonomy's attribution rule: label the earliest failure from which execution does not
  recover;
- gives a definition in the style of the existing ones;
- explains why the closest existing modes misclassify these failures;
- includes at least two independent, synthetic or redacted trace excerpts, plus a similar
  case that should not get the new label.

To try a mode before it is accepted, copy `taxonomy_v1.json`, add your mode and pass the file
with `--taxonomy FILE`.

## Troubleshooting

- **"raw source task does not match the rollout"**: the source data is
  inconsistent. Use another run and report the data issue. Do not work around
  the check.
- **Files skipped in a report**: empty, corrupt, or unrelated JSONL files are
  skipped and listed with the reason. Results from a different taxonomy or
  protocol are listed too, but counted separately.

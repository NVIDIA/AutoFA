# Evaluating unassisted discovery

Deterministic regressions exercise native family, coverage, diagnostic and repair machinery.
They do not prove that a discovery model independently chooses a useful hypothesis. The
optional `tools/autofa_autonomy_eval.py` harness prepares baseline and updated arms from the
same raw dataset, explicit allowed profiles, budgets and seeds. It supplies no detector,
selected example, repair code or added discovery prompt. No model calls occur during prepare.

Create a JSON settings file with your permitted harness/model/API configuration,
`classifier_profiles` (explicitly `[]` if escalation is prohibited), `max_rounds`, and relevant
native comparison limits. For a controlled ablation, include all three objective fields:

```json
{
  "max_rounds": 8,
  "classifier_profiles": [],
  "classifier_comparison_limit": 3,
  "objective_axes": ["ppv_at_pi0", "task_recall", "behavior_purity"],
  "objective_weights": [20, 50, 30],
  "objective_rank": "geometric_mean"
}
```

Add the same explicitly approved model/provider settings for both arms. Keep credentials in
environment variables, not this file. Preparation rejects literal credential/header fields
and keys that are not `Config` fields in both checkouts before writing artifacts;
environment-variable routing names and generation controls remain valid.
For a default-experience comparison, omit objective
fields and use `--mode default-experience`; compare the recorded effective settings rather than
attributing every difference to one code change. Prepare each input dataset as its own experiment:

```sh
python tools/autofa_autonomy_eval.py prepare --output /tmp/autofa-comparison \
  --baseline /path/to/baseline --updated /path/to/updated \
  --source /path/to/raw-data --settings /path/to/permitted-settings.json \
  --seeds 17 29 43 --timeout-seconds 7200 --mode controlled
```

Preparation records exact runtime/source hashes and creates separate homes/settings for each
arm. Use a Python environment with dependencies for both checkouts. The following command
**runs real native discovery and incurs model usage**; run it only within the authorized data,
model and spending scope:

```sh
python tools/autofa_autonomy_eval.py execute /tmp/autofa-comparison/manifest.json \
  --arm updated --seed 17 --python /path/to/python
```

Run each prepared arm once. The harness refuses changed source/code/settings and repeated
execution. It disables automatic target-driven round extension and publishing; the declared
wall-clock limit interrupts the owned process group, retaining logs and native checkpoints.
Timeouts and failures are outcomes, not permission to reroll. External service requests may
already have incurred usage before interruption. Inspect `execution.json`, `effective-settings.json`,
`native-result.json`, `native.log`, and the arm's `home/runs/` artifacts.

For each arm, record native candidate origins, definitions and edit history, source inspections,
family synthesis, classifier comparisons, retained controls, confirmed versus sampled grades,
failed-attempt union and task coverage with denominators, per-benchmark coverage, unknowns,
errors, runtime and measured usage. Keep unmeasured costs unknown. Native coverage-filtered
sampling balances multiple selected benchmarks before its bounded row read; it does not imply
source-run balancing when sources share a benchmark. Explicit `by_benchmark=False` retains
pooled sampling.

An arm that receives an external detector, example-specific lead or repair is assisted and
cannot establish unassisted success. Preserve that intervention and report its continuation
separately. Review failed/recovered intermediate reasoning and wrong delivered answers under
the saved judge policy. Use frozen additional evaluation/source audits and fresh related data
when available; the prior campaign's detector names and reported scores are not gold labels.
Preparation, mocked model outputs and transport fidelity tests establish no real-world recall
improvement. A paid live comparison is still required to measure that claim.

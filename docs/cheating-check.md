# Preflight: cheating

`autofa preflight cheating` finds rollouts where the agent gamed the evaluation instead of
solving the task, such as looking up the reference fix or editing the tests, and
writes an offline HTML report.

After [setup](../README.md#quick-start), prefix the commands below with
`uv run --python 3.12` when running from the checkout.

```bash
autofa preflight cheating                                   # pick runs and a review count interactively
autofa preflight cheating --api-run RUN_ID                  # a remote run
autofa preflight cheating --input ./eval-results            # a local NeMo-Gym / Harbor run
autofa preflight cheating --api-run RUN_ID --screen-only    # patterns only, no judge
```

In **New analysis** or **Continue from library**, open **Analysis settings > Analyses to run > Preflight >
Cheating** to enable it for that run. Use **Settings > Analyses to run > Preflight >
Cheating** to save the default for subsequent runs. **Efficiency metrics** defaults to
**Off**, **Failure discovery** to **On**, and **Cheating** to **Off**.
Enable cheating first to turn both other analyses off and run only preflight.

When enabled, cheating runs on the selected dataset before analysis, using the
same model and benchmark filters. It screens passed and failed rollouts and
reviews all flagged rollouts. Findings are reported without removing rollouts
or stopping analysis; a failed judge setup stops the run with an error. Resume
uses the run's saved preflight-check selection and reuses successful cached judgments.
When the check finishes, the cheating report opens in your browser, from the standalone
command and in analysis runs; on Linux without a display, only its link is printed.
The standalone command above runs cheating regardless of the saved toggle and
offers review limits and screening without an LLM.

## What gets checked

Every rollout is checked, passed or failed. Runs and rollouts are selected with
the same flags as `autofa prepare` (`--api-run`, `--input`, `--model`,
`--benchmark`, `--task`, `--latest`, `--per-run`, ...).

- Cheap patterns scan each trace for marks of cheating (git history lookups,
  fetching the fix, editing existing tests, ...). Rollouts without a hit count
  as clean and are not reviewed. A test edit counts when it changes or removes a
  line that was there before, whether made with an editor, a shell command,
  `git apply` or `apply_patch`; adding lines does not, so a test disabled by an
  added early `return` is not caught (added skip markers are, by their own pattern).
- An LLM judge reviews every flagged rollout against
  [the rubric](../src/autofa/preflight/cheating/rubric.md), without seeing whether
  it passed. A "no cheating" verdict counts only if the judge read every kind
  of pattern hit; otherwise the rollout is listed as needing attention.
  `--max-reviews N` reviews a random sample of N instead; the
  interactive flow asks for this number (10, 50, all, or custom).
- `--reward '>=0.5'` filters by numeric reward. Datasets whose `is_correct` or
  `accuracy` does not match that reward threshold should pass the appropriate
  reward comparison explicitly.

The rubric covers agentic coding tasks (SWE-bench, Terminal-Bench style). On
other tasks the judge gives no verdict.

## Judge

Reviews use the HTTP endpoint configured by `judge_*`, inheriting the detector
URL/key when overrides are empty, with the `cheating_check_model` setting.
Configure the shared endpoint in **Settings > LLM endpoints > Detector LLM endpoint**;
**Judge endpoint** instead selects the Claude Code/Codex behavior judge.
`--judge-model` picks another model and
`--judge-api-key-env NAME` another stored key. Before the first review, the
command prints how many reviews will run and the judge tokens they took on
earlier runs. While they run, it shows how many are done, the judge tokens
spent and the time left at the pace so far. At the end, one line says "Cheating
found", "No cheating found" or "Not settled", with how many flagged rollouts stay
open; "No cheating found" appears only when every flagged rollout was settled.
Successful verdicts are reused for unchanged rollout content,
rubric, screening version/patterns, and judge model. Endpoint access checks can
still make requests.

## Outputs

`cheating_report.html` and `cheating_check.json` are written to
`$AUTOFA_HOME/preflight/cheating/<run>-<time>/`, or `--output`. For an analysis
run they are saved in `<analysis-run>/preflight/cheating/`. Each rollout links to
its trace in the EvalRDL viewer. For local runs checked with `--input`, the report
offers a link to copy instead: browsers can't open the Local Library server's
traces from a file, so paste it into a new tab while AutoFA is running.

The report counts, each a part of the one before, with its share of the reviewed
rollouts (flagged rollouts show their share of all rollouts):

- **Cheated:** the judge found cheating of any kind, including attempts that
  failed or could not change the result.
- **Could have caused a pass:** the kind of cheating that can make a task pass
  (used the reference fix, edited tests with effect, ...), whether or not this
  rollout passed.
- **Contaminated pass:** the rollout passed and its cheating could have caused
  it, so this pass can't be trusted.

The runs table lists each run's behaviours. Below it, the report breaks cheated
rollouts down by behaviour; click one to list only those rollouts. A folded list
above it says what each behaviour means.

The check does not test whether the cheating actually changed the score.

## Adding another preflight check

Each check lives in its own package under `src/autofa/preflight/`. Register
a `PreflightDetector` in `preflight/registry.py` with its name, label, description
and entrypoint module. That module exposes `main(argv)` for its standalone CLI
and `run_dataset(dataset, cfg, output, *, sources)` returning a `PreflightResult`.
The supplied dataset already carries the analysis filters; return report and
data paths plus a nonzero status for execution failures, not positive findings.

The shared registry supplies the Preflight submenu, `autofa preflight <check>`
commands and analysis runner. Additional checks default to Off and do not
need branches in the home menu or analysis lifecycle. The cheating package owns
its screening patterns, rubric, judge pipeline, report and cache. Judgments
are cached under `$AUTOFA_HOME/preflight/cheating/`.

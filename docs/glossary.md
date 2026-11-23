# Glossary

[Back to README](../README.md) · [Running an analysis](workflows.md) · [Detectors](detectors.md)

AutoFA's documentation, user interface and agent instructions use the terms below.
Each entry gives the preferred term, what it means, and any terms to avoid in its place.
Configuration keys, CLI flags and API names keep their existing spelling even where
the prose term differs; for example, the efficiency toggle is `--analysis-output efficiency`
and discovery is `--analysis-output agent`.

## Workflows

AutoFA is a failure-analysis toolkit. Each **workflow** answers a different question about
the same evaluation runs.

| Term | Meaning | Avoid |
|---|---|---|
| **Workflow** | One of AutoFA's top-level analyses: preflight checks, failure discovery, efficiency metrics and taxonomy classification. In run-config files, `workflow: classification` selects taxonomy classification; the default `workflow: discovery` covers failure discovery, efficiency metrics and preflight checks, chosen with `analysis_outputs` and `preflight_detectors`. | analysis mode, pipeline |
| **Failure analysis** | Everything AutoFA does to explain why rollouts fail (the *FA* in AutoFA). An umbrella term, not the name of a workflow. | using it to mean failure discovery alone |
| **Preflight check** | An optional check that runs before the other workflows. **Cheating** is the current preflight check: it flags rollouts where the agent gamed the evaluation. | preflight detector |
| **Failure discovery** | The bottom-up workflow. An agent reads failed rollouts, proposes failure modes and writes [detectors](#detectors) for them, then AutoFA scores the detectors. Short form: *discovery*. | detector discovery, discovery loop, AutoFA failure analysis, autodiscovery |
| **Efficiency metrics** | The workflow that compares tokens, context, turns, tool use and cost between correct and incorrect rollouts. It runs alone or alongside failure discovery and writes an **efficiency report**. | efficiency analysis (as the workflow name) |
| **Taxonomy classification** | The top-down workflow. A judge gives each failed rollout exactly one failure mode from a fixed taxonomy. Short form: *classification*. | taxonomy workflow |
| **Analyses to run** | The menu under **Analysis settings** (per run) and **Settings** (defaults) that turns **Efficiency metrics**, **Failure discovery** and **Preflight** checks on or off. The UI calls each of these an *analysis*. | |
| **Analysis target** | The discovery setting that steers it toward model faults, harness faults or both (`analysis_focus`). | analysis focus |
| **Pass setup** | The combination of first and second taxonomy passes, each LLM or agentic, such as *LLM + agentic review*. | workflow |
| **Taxonomy** | A fixed tree of failure modes used by taxonomy classification. The built-in taxonomy separates model faults from harness faults. | ontology |

Compound names such as *repair workflow*, *family workflow* or the GitHub Actions *CI workflow*
describe internal procedures, not AutoFA workflows.

Failure discovery and taxonomy classification are complementary: discovery finds
failure modes in the data, while classification maps failures onto known categories. A
taxonomy bucket can seed a discovery run. Inside discovery, **top-down** and **bottom-up**
also name two ways of grouping detectors into [families](#detectors); that is a separate,
discovery-internal use of the words.

## Inputs

| Term | Meaning | Avoid |
|---|---|---|
| **Evaluation run** | A completed benchmark run that AutoFA reads: a local NeMo Gym or Harbor folder, or a remote EvalRDL run. AutoFA does not launch evaluations. | eval run, source run |
| **Task** | One benchmark problem. A task can have several rollouts. | datapoint |
| **Rollout** | One agent attempt at one task, with its trace, reward and grader evidence. A **failed rollout** is a healthy, scored rollout with `reward <= 0` by default; taxonomy classification accepts other thresholds through `--reward`, and remote runs use the outcome labels their API reports. | episode, trajectory (for the whole rollout), sample, example |
| **Attempt** | Used only in metric names: *attempt-level* metrics and *attempt recall* count each rollout separately, while *task-level* metrics count each task once. | attempt as a general synonym of rollout |
| **Trace** | The recorded events of a rollout: messages, tool calls, tool results and their metadata. Tools and fields that expose its parsed step sequence say *trajectory*, as in `get_trajectory` and `trajectory_steps`. | transcript, log (for a single rollout) |
| **Item** | The object a detector receives for one rollout in `detect(item)`. | |
| **Dataset** | The rollouts that one analysis run reads after source preparation. Scoring terms call it the *corpus*: a **full-corpus scan** scores every rollout in the dataset instead of a sample. | |
| **Selection** | A saved set of rollouts chosen with filters such as `--per-run` or `--reward`; reusable across workflows. | |
| **Cohort** | The rollouts a discovery run scores detectors on, such as failures plus successful controls. | |

## Detectors

| Term | Meaning | Avoid |
|---|---|---|
| **Failure mode** | A recurring reason why rollouts fail, such as *retries the same failing command instead of diagnosing the error*. It has a short **description** and a **mechanism**. | failure pattern, failure category, failure type |
| **Mechanism** | The causal explanation of a failure mode: why the agent fails, not just what it does. | |
| **Detector** | Python code, `detect(item) -> bool`, that flags rollouts showing one failure mode, together with its description, mechanism and scores. Detectors can combine regular expressions, parsing and LLM calls. | function, labeling function, classifier, rule, checker |
| **LLM detector** | A detector that calls the **detector LLM endpoint** for part of its decision. | API detector, LLM-based detector |
| **Helper** | Reusable, unscored Python code that detectors and other helpers call, for example to parse or extract passages. Helpers make no failure claim. | |
| **Detector family** | A parent detector and its narrower children, built during discovery. | failure family |
| **Candidate** | A detector that discovery has written and is still evaluating, or a cloud detector proposed for import. | hypothesis (for the code) |
| **Coverage** | Share of failed rollouts that a detector flags; the report name for `recall`. | |
| **Detector adherence** | How well a detector's flagged rollouts support its full claim, as reviewed by the behavior judge; the report name for `behavior_purity`. | behavior-purity score, causal-review adherence |
| **Causal review** | The behavior judge's review of one matched rollout: does the evidence show the claimed mechanism caused the failure? Discovery tracks confirmed coverage separately from raw matches. | |
| **Judge policy** | The review standard the behavior judge applies, set in **Analysis settings > Judge policy** (`behavior_strictness`). | |
| **Failure card** | The report entry for one detector: its failure mode, mechanism, *How the detector works*, scores and evidence. | detector card |

See [metrics and feedback](metrics.md) for every score.

## Results

| Term | Meaning | Avoid |
|---|---|---|
| **Analysis run** | One AutoFA execution with its own run ID, saved under `~/.autofa/runs/<run-id>/`. | AutoFA run, when it could be confused with an evaluation run |
| **Session** | The conversation in which the AutoFA harness carries out discovery for an analysis run. Resume continues it. | |
| **Round** | One discovery cycle of sampling, proposing and scoring detectors. Limits such as `round_limit` count rounds. | iteration |
| **Campaign** | A long discovery effort over several rounds or analysis runs, usually supervised by the driver, with agreed targets and limits. | |
| **Failure library** | The saved detectors, helpers and reports of your analyses, browsable in the local **Library** (default <http://127.0.0.1:8765>). Short form: *library*. Some screens call a saved library a *catalog*. | detector library |
| **Remote library** | A failure library published to **AutoFA Cloud** (a *cloud library*) or to a self-hosted AutoFA server, so that others can browse it and reuse its detectors. CLI flags such as `--remote-library` use this name. | |
| **Report** | An offline HTML file written by a workflow: the **Model Failure Cards** report, the efficiency report, the taxonomy report or the cheating report. | |

## Agents and models

| Term | Meaning | Avoid |
|---|---|---|
| **Host agent** | Your coding agent, such as Claude Code or Codex, when it operates AutoFA for you. | outer agent |
| **Driver** | The `autofa-driver` skill that the host agent uses to set up, launch, monitor and explain workflows. | supervisor (for the skill) |
| **AutoFA harness** | The Claude Code or Codex CLI that AutoFA runs as its **discovery agent**. Set it in **Settings > LLM endpoints > AutoFA harness**. | |
| **Agent harness** | The harness that produced the evaluation runs under analysis, such as Codex, OpenCode or LangChain. Say *evaluated harness* when it could be confused with the AutoFA harness. | |
| **Behavior judge** | The harness session that checks whether a detector's matches support its claimed mechanism. It uses the **Judge endpoint** settings. | behavior-purity judge |
| **Taxonomy judge** | The judge in each taxonomy pass: the HTTP model in `judge_model` for LLM passes, or the Judge endpoint harness for agentic passes. | |
| **Investigation call** | A tool call in an agentic taxonomy pass that can reveal new evidence or taxonomy information, such as reading, searching or listing. The per-pass budget (`taxonomy_agentic_max_turns`, `--agentic-max-turns`) counts only these calls; citation, review-status and submission calls are free. | turn, step (for this budget) |
| **Detector LLM endpoint** | The OpenAI-compatible endpoint that LLM detectors call. Set it in **Settings > LLM endpoints > Detector LLM endpoint**. Commands such as `autofa preflight classifiers` call the model behind it a *classifier*. | feature model (outside configuration keys) |
| **Improvement classifier** | An auxiliary LLM call that, after judging, suggests whether the model, the harness or both could fix a failure mode. It steers the discovery agent and is saved in the library; it does not change detector scores. See [Improvement targets](configuration.md#improvement-targets). | |
| **Readiness check** | Validation that runs before paid work, such as harness login and endpoint checks. It is separate from preflight checks, even though some readiness tools live under `autofa preflight`. | preflight (for readiness validation) |

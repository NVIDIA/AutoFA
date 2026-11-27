# Configuration and customization

[Back to README](../README.md) · [LLM endpoints](llm-endpoints.md) · [Workflows](workflows.md)

[Credentials and harness selection](#credentials-and-harness-selection) · [Configuration (autofa.toml)](#configuration-autofatoml) · [Analysis target and taxonomy](#analysis-target-and-taxonomy)

## Credentials and harness selection

Use the [quick start](../README.md#quick-start) to install and launch. Sign in with the CLI selected
in **Settings > LLM endpoints > AutoFA harness**: `codex login` or
`claude auth login`. Make its executable available on `PATH`, or set
`AUTOFA_CODEX_CLI` / `AUTOFA_CLAUDE_CLI` to its path. Claude also supports
`ANTHROPIC_API_KEY` with the binary bundled in `claude-agent-sdk`, or a sandbox's
`CLAUDE_CODE_EXECPATH`. The behavior judge uses its selected harness's credentials;
LLM detectors use the separately configured [LLM endpoint](llm-endpoints.md#llm-endpoints).

For Codex behavioral reviews on hosts where creating a fresh SQLite index times out,
set `AUTOFA_CODEX_EPHEMERAL_REVIEW=1` for the AutoFA process. This starts a fresh
[ephemeral Codex thread](glossary.md#agents-and-models) for each review and verifies
that the server acknowledged both ephemeral mode and an empty tool environment before
submitting evidence.
AutoFA retains its own review receipts. A bounded follow-up can continue the
same live review thread. This mode uses Codex's configured SQLite state directory.
The server acknowledgement describes the review thread's persistence, and the shared
directory can still contain Codex operational state and diagnostics.

Save keys in the project's `.env`, or share them across checkouts through
`~/.autofa/.env` (`$AUTOFA_HOME/.env` for a separate profile):

```dotenv
LLM_API_KEY=your-api-key  # The variable your detector endpoint is configured to read
# Optional Claude API-key authentication:
# ANTHROPIC_API_KEY=your-anthropic-key
```

Restart AutoFA after changing `.env`. Existing environment variables take precedence,
followed by the nearest project `.env`, parent project files, and the shared file.
Keep credentials out of Git; restrict the shared file with `chmod 600 ~/.autofa/.env`.
AutoFA has no built-in detector endpoint. Use the variable name you selected for
your [LLM endpoint](llm-endpoints.md#llm-endpoints).

Interactive discovery and Resume check detector inference access. By default
(`require_llm_detectors = true`), a missing endpoint, missing credentials or failed checks
stop discovery before analysis. Set `require_llm_detectors = false` in `autofa.toml` to
continue with a warning; deterministic detectors can still run. LLM evaluation errors remain
explicit instead of becoming negative labels. Batch YAML can also require a working detector
endpoint with `require_llm_detectors: true` under `execution`; it cannot turn off a
requirement set in `autofa.toml`.

Saved settings take priority. On first launch, AutoFA checks `codex login status` and `claude auth status`, prefers authenticated
Codex, and falls back to authenticated Claude Code. If neither is available, that
automatic setup exits with sign-in instructions without saving a config. Checks
have a five-second timeout per CLI and do not verify model access or quotas.
`autofa --help` is available without a configured harness.

## Configuration (`autofa.toml`)

AutoFA uses a project-local `autofa.toml` by default. On the first launch in a
checkout, existing `~/.autofa/config.toml` settings are copied into it without
deleting the original. An existing project config always wins; later home-file
edits are not synchronized into it. Settings autosaves to the active file, and
the CLI and discovery agent context identify that file's actual path.

Consequently, downloading a fresh checkout does not necessarily select fresh defaults.
For current defaults, use a new, nonexistent profile path through `AUTOFA_CONFIG` before
the first configuration-creating command. This bypasses the implicit home-file copy;
the native initializer applies the installed version's defaults and deployment settings.
Existing explicitly selected files keep their saved choices. The driver records the
selected file's origin and any intentional overrides during setup.

Path precedence: explicit `AUTOFA_CONFIG` file; explicit `AUTOFA_HOME/config.toml`
profile; nearest existing project `autofa.toml`; AutoFA's source checkout;
the caller's Git/project root; home config only when no project is available.
Relative taxonomy, skills, additional Markdown, and library paths are resolved
beside the config file. `disabled_skill_paths` stays relative to the skills folder.

The local `autofa.toml` is git-ignored to avoid committing personal paths or private
extensions. API keys still belong in `.env` or the credential store, not config.
Runs, caches, and the durable library remain under `~/.autofa/` unless `AUTOFA_HOME`
is explicitly set. Moving the config does not move or discard these artifacts.

Created on first launch. Edit and restart AutoFA. On startup, AutoFA fills in
missing settings without overwriting your saved choices.
Unrecognized keys in `autofa.toml` are ignored; keys left by earlier releases
(for example `min_failure_coverage` or `pi0`) can be deleted.

For discovery setup, `autofa config inspect --run-config RUN.yaml --json` resolves the exact
models, objective, limits and publication choice without inference, authentication checks or
configuration writes. It identifies inherited files and missing monetary rates. After agreement,
`autofa config prepare --run-config RUN.yaml --expected AGREED.json --output NEW_DIRECTORY`
freezes that setup into a scoped profile and batch file. The returned managed-launch specification
pins their hashes and sets `AUTOFA_SETUP_CONTRACT`; native execution verifies the agreement
before readiness checks and again after run creation. Resume and approved live changes keep their saved
settings. See [driver model setup](../agent-skills/autofa-driver/references/model-setup.md).

**Terminate if stalled** is On by default and applies only after the selected planned
rounds. **Stalled extra rounds** (`discovery_stall_rounds`, default `3`) counts consecutive
completed recovery rounds without meaningful validated library improvement. The counter
starts at zero after planned rounds; each extra round includes investigation, validation,
and its bounded closing refinement exchange. Repeated failed probes trigger recovery
guidance rather than terminate a planned round. Optional coverage-plateau exploration uses
`discovery_plateau_patience` (default `6`). This is independent of the validated recovery-round
counter.
Resume preserves the run's saved round patience, while Continue uses the newly selected
settings. Disabling stall termination leaves the existing round and detector targets in control.

An exhausted review allowance is a separate operational limit. When even qualifying every
remaining unblocked candidate in a frozen polishing library would fall short of the requested
minimum, AutoFA stops with `review_capacity_exhausted`. It saves incomplete
results and the review-processing reason without counting a completed no-improvement round,
rejecting the detector, or resetting its review budget. This block also applies when quality
stall termination is disabled. Unchanged Resume cannot replenish a consumed allowance;
inspect the saved review workload and evidence before arranging recovery.

Settings → **Judge policy** selects how detector claims are reviewed:
**Behavior match** checks that the described behavior occurred, including any claimed causes.
**Failure relevance** also requires a trace-supported link to failure and checks
later recovery. **Causal contribution** (default) requires evidence that the behavior contributed
to the task failure and was not recovered. Discovery,
judging, repairs, ranking and coverage use the same selection. Resume keeps the saved
policy; Continue uses the newly selected policy and retains older reviews as history.
In the configuration file, these correspond to `behavior_strictness` values
`behavioral`, `contextual`, and `causal`. The precision/recall optimization preset is
independent of the judge policy.

Base defaults live in [`src/autofa/defaults.toml`](../src/autofa/defaults.toml).
No detector endpoint is built in, and remote services are unset.
Saved `autofa.toml` values take priority, including explicitly blank service URLs.
`AUTOFA_SITE_CONFIG` can select an optional deployment preset, which may provide a
shared detector endpoint through `hub_base_url` and `hub_api_key_env`. Without it,
a `config/internal.toml` beside the source is used; Git ignores that file. Configure remote
services through **Settings > EvalRDL remote server** and
**Settings > AutoFA Cloud / library server**; a blank EvalRDL URL hides remote
dataset selection, and a blank library URL disables remote publication.

Python uses these defaults to create `autofa.toml` and fill missing settings.
Edit your local `autofa.toml` to customize your setup. The
`[harness_defaults.claude]` and `[harness_defaults.codex]` tables set the initial
model and reasoning controls during first-launch detection and when switching
harnesses. Saved configurations retain their choices.
`campaign_policy` separates a discovery time/spend horizon from checkpoint rounds and is
saved with the run. The default finishes on target: discovery stops once the Good minimum is met
and confirmed detectors cover 80% of failed rollouts, and never runs past round 5; every third
round rescores the full dataset. Set `campaign_policy = {}` to stop on rounds and the minimum
alone. `usage_pricing` snapshots agreed API rates before campaign startup calls and defaults to
an empty object. See [campaign controls](driver-supervision.md#time-and-spending-horizons)
for policy fields, estimated spending limits, bounded overtime and live `supervise update`.
`require_llm_detectors = true`, the default, requires a successful detector endpoint readiness
check before discovery. Campaign startup accounts for that probe and obeys admission limits; a failed or
blocked required probe cannot produce a successful empty batch. Existing saved findings remain
available for partial delivery. Batch YAML also exposes `execution.require_llm_detectors`.
The example below shows the base defaults with Codex selected:

```toml
harness = "codex"
top_x = 5
max_rounds = 5
min_failure_modes = 4  # Supported Good detectors; continue until met. 0 disables.
campaign_policy = { mode = "finish_on_target", round_limit = 5, coverage_target = 0.8, coverage_unit = "failed_attempts", checkpoint_every_rounds = 3 }
behavior_strictness = "causal"  # Judge policy: Causal contribution.
discovery_wait_for_plateau = false  # Regular discovery: optional extra coverage search.
model = "gpt-6-astra"
thinking = "default"  # Let the discovery harness choose.
effort = "default"
thinking_budget_tokens = 10000  # Used for budgeted thinking only.
behavior_judge_harness = "auto"
behavior_judge_model = "gpt-6.1-sol"
behavior_judge_thinking = "default"  # Claude-only; Codex uses effort.
behavior_judge_effort = "medium"
behavior_judge_thinking_budget_tokens = 10000
judge_workers = 4  # Independent behavior-judge examples per batch; integer 1-8.
llm_eval_workers = 8  # Requested LLM detector concurrency; integer 1-256.
llm_eval_max_workers = 128  # Ceiling for LLM detector concurrency; integer 1-256.
llm_eval_queue_mib = 64  # Buffered scoring input limit in MiB; integer 1-4096.
llm_eval_sample = 200  # Stratified LLM detector sample during capped evaluation.

feature_model = "openai/openai/gpt-6-luna"
judge_model = "openai/openai/gpt-6-luna"
feature_api = "auto"  # prefer Decisions; overrides: responses, decisions, chat_completions
judge_api = ""  # inherit feature_api
hub_base_url = ""  # No built-in endpoint; a deployment preset may set one
hub_api_key_env = ""
taxonomy_first_pass = "llm"  # llm or agentic
taxonomy_second_pass = "llm"  # off, llm, or agentic
taxonomy_agentic_max_turns = 128  # Investigation-tool calls per agentic pass

# Optional EvalRDL remote server (run catalog + AutoFA parquet export)
data_api_base_url = ""

# Optional remote library service. Credentials use env or OS keychain.
remote_library_base_url = ""
remote_library_token_env = "AUTOFA_LIBRARY_TOKEN"

# --- Detector field policy (anti-cheat / portability) ---
# content  = only prompt / CoT / answer text (DEFAULT)
# metadata = allow judge_text / scoring JSON (labels still banned)
# custom   = use allowed_fields below
detector_fields = "content"
allow_metadata = false
# allowed_fields = ["prompt_text", "response_text", "chain_of_thought"]
# extra_allowed_fields = ["scratchpad"]

# --- Optimization objective (Pareto axes to maximize) ---
objective_axes = ["ppv_at_pi0", "task_recall", "behavior_purity"]
objective_rank = "geometric_mean"
objective_weights = [20, 50, 30] # Recall preset; explicit positional weights follow objective_axes
# Precision: [50, 20, 30]; Recall: [20, 50, 30]

# --- Creative exploration (String Seed of Thought — arXiv:2510.21150) ---
# Encourages diverse hypotheses using a seeded exploration string. Default on.
exploration_ssot = true
# Fixed main seed; "auto" chooses a different main seed for each fresh run.
exploration_seed = 42
# false = open-ended (default); true = soft mechanism-template priors
exploration_templates = false
```

Capacity settings are validated when configuration is loaded. `llm_eval_workers` and
`llm_eval_max_workers` accept integers from 1 to 256; `llm_eval_queue_mib` accepts 1 to 4096;
`judge_workers` accepts 1 to 8. Effective LLM concurrency is the smaller of the request and
ceiling. Per-run changes are available through `autofa supervise configure`.

Always edit `objective_axes` and `objective_weights` together: weights are positional.
Reordering axes requires reordering their weights, and removing an axis requires removing
its weight. Set `objective_weights = []` for equal weighting.

**Why content-only by default?**
If detectors read harness grader fields (`judge_text`, `hb_scoring_details_json`, test status, …), they “cheat” and will not transfer to other data. Outcome labels (`is_correct`, …) are always banned.

**Creative exploration (SSoT):** When `exploration_ssot = true` (default), the agent uses a
string derived from the current round's seed once at the start of that round to choose an
evidence-supported investigation focus ([String Seed of Thought](https://arxiv.org/abs/2510.21150)).
At a round change, it uses the previous findings and the new string to consider a useful
pivot. Within the round, evidence guides candidate choices and repairs; repeated kickoffs
retain the focus. Overlapping detectors and subsets can add useful explanations or remedies.
Discovery stays open-ended (`exploration_templates = false`); set it to `true` to opt into
soft mechanism-template priors.

The default `exploration_seed = 42` is the **main seed** for a reproducible sequence of
different discovery-round seeds. `sample_items()` automatically uses the current round's
seed, so the next sampling call after advancing a round selects different examples without
asking the agent to invent seeds. Examples enter context through MCP tool responses; this
does not automatically insert a dataset sample into the round's initial prompt.
With no count arguments, `sample_items()` requests 12 failed rollouts and 2 successful
rollouts (or fewer when the eligible pool is smaller). Explicit counts override these
defaults; sample size does not change automatically between rounds.
The exploration string varies with that round too. The same main seed, dataset and sampling
requests reproduce the same sequence across fresh runs. Samples can overlap; distinct-task
sampling still prevents repeated failed rollouts within a sample.

Resume retains the current sequence position. Continue from a saved or published library
inherits its main seed and starts at the next position, rather than replaying the first
round. The main seed and sequence position travel in the library. A library without them,
such as an efficiency-only publication, starts after its last recorded detector round (at
least position 1).

The **200-example iterative scoring panel stays fixed** across discovery rounds and Continue
for the same corpus and main seed. Set `exploration_seed = "auto"` to choose a different main
seed for each fresh run. `exploration_ssot = false` disables only the string prompt; discovery
sampling still advances through the seed sequence. Explicit sampling seeds remain available
to replay a particular slice or explore further within a round: `sample_items(seed=123)`
overrides only that call, leaving the main seed and future round sequence unchanged.

For comparisons, keep the dataset snapshot/order, code, model versions, harness, reasoning
settings and experiment budgets the same. A fixed seed reduces input variation; agent
decisions, generated detectors and LLM judgments can still differ.

### Optional classifier comparisons during discovery

Ordered fallback classifiers let the discovery agent investigate whether its detector
model can execute an evidence-supported predicate. These are detector classifiers,
separate from the discovery agent and the behavior judge. By default, AutoFA permits a
ladder that raises the Luna baseline to medium reasoning, then moves to Sol at medium and
high reasoning:

```toml
classifier_comparison_limit = 3
classifier_profiles = [
  { name = "luna_medium", feature_model = "openai/openai/gpt-6-luna", feature_api = "responses", feature_request_options = { reasoning_effort = "medium" } },
  { name = "sol_medium", feature_model = "azure/openai/gpt-6.1-sol", feature_api = "responses", feature_request_options = { reasoning_effort = "medium" } },
  { name = "sol_high", feature_model = "azure/openai/gpt-6.1-sol", feature_api = "responses", feature_request_options = { reasoning_effort = "high" } },
]
```

The default profiles inherit your detector endpoint and credentials, so that endpoint must
serve both deployments. Replace the model IDs with your provider's deployments, or set
`classifier_profiles = []` to disable alternative-model calls. Configured profiles also make
the detector readiness check required. An incomplete inherited endpoint is reported when
AutoFA creates a detector client, not when it loads the settings.

Each profile requires a unique `name` and `feature_model`. It may override
`feature_api`, `feature_base_url`, `feature_api_key_env`, and
`feature_request_options`. Omitted fields inherit the run's detector settings when
its comparison policy is first saved, rather than overrides from a preceding fallback.
Use only models and request options your provider supports. Credentials remain in the
environment variable referenced by `feature_api_key_env`.

`judge_api = ""` inherits `feature_api`; set it to `auto`, `decisions`, `responses`,
or `chat_completions` to choose the judge API explicitly. When AutoFA adopts a detector
profile, it saves the judge's previously inherited base URL, API-key environment variable,
and API choice for that run. Detector escalation therefore preserves the judge and
improvement-classifier routes. Explicit judge settings still take precedence.

The agent uses `classifier_profiles(status|compare|adopt)` inside a candidate
investigation. A comparison runs identical source and 2–30 fixed controls with both
the current classifier and the next untested approved profile. A failed comparison advances
trial history without adopting its profile, so a later approved tier remains available.
Controls include declared
positive and negative expectations. Their results are diagnostic, not independent
validation or population scores. A passing baseline skips fallback inference.

The comparison limit is per run and survives interruption and round changes; the
default is three comparisons with at most 120 SDK dispatches per comparison across
both classifiers. AutoFA retries consume this allowance; provider SDK transport
retries remain inside the client. Repeated identical requests reuse saved results.
Source, control evidence, expectations, dependencies and classifier settings are
tracked. Expectations for the same source, input controls and classifier pair are
fixed from the start of a comparison, including interrupted comparisons.

Adoption requires the fallback to pass every declared control without errors, improve
at least one positive or negative Boolean control, and preserve every baseline-passing
result. Declared abstention controls must remain abstentions. Adoption selects the next
profile for this run and preserves saved scores and historical evidence. Ordinary
evaluation uses the selected profile; loading or publishing never invalidates scores. The global configuration is unchanged.
Compare/adopt are unavailable during final polishing or after run completion. A
configured profile does not guarantee better recall or causal quality.

## Analysis target and taxonomy

Open **Settings** from the home menu (or run `autofa settings`) to choose a
persistent analysis target, manage Markdown skills, or change the
optimization weights. Each valid change is saved automatically; Back closes the menu.
The default analysis target is **Model + harness** (stored as `both`).

Interactive New and Continue first show **Analysis settings**, with an **Analyses to run**
submenu and controls for the target and skills. Changes apply only to that
run. **Continue** is selected first. Back or Esc/q in a submenu returns to the setup
screen with your selections; Esc/q from that screen discards the run's changes. Explicit
guidance flags and non-interactive launches skip this extra menu.

**Settings > Optimization weights** has one slider per objective, adjusted with
left/right; up/down selects a row. The three standard objectives are standardized
precision, distinct-task recall, and behavioral purity. Presets are Balanced (equal thirds),
Precision (50/20/30), and Recall (20/50/30), in that order. Recall is the default. Its family-comparison allowance is larger than Balanced’s;
explicit round, model and experiment budgets remain in force.
Weights are normalized; zero disables an axis and all-zero is rejected. The weighted
geometric mean gives zero if an active metric is zero. Positive weights do not change
Pareto dominance. Search scores use these preferences; the live search trend shows the
best current comparable score, alongside grade counts and assessed confirmed task coverage.
The historical corpus-frontier average remains a separate diagnostic. These preferences
are not calibrated quality thresholds and do not change the judge policy.

Discovery guidance targets **Outstanding** detectors. Public CLI/report grades are capped
at **Good**: Coverage >1%, normalized precision ≥0.70, adherence ≥0.90, and support
across at least five distinct matched failed tasks. Higher Coverage remains the exploration goal.
The CLI and report use that quality order. Configured scores and hypervolume remain
progress diagnostics; their calculations and final validation requirements are unchanged.

The preset also guides discovery and judge expectations for explanation breadth:

- **Precision:** prefer accurately detected, specific mechanisms and sharp boundaries.
- **Balanced:** seek useful causal families with informative child mechanisms; alternate
  investigating uncovered failures with checking candidate boundaries.
- **Recall:** generalize incidental task details and seek causal families covering more
  distinct failures, retaining specific children when useful.

Every optimization preset retains the selected judge policy. The default Causal contribution
policy requires an unrecovered contribution to the task failure; Failure relevance requires
a trace-supported connection without demanding the full causal chain. Broad families may
contain different technical subtypes. Specificity describes explanatory detail and cannot
by itself reject a supported family. Preset changes do not change the judge policy or
invalidate saved reviews; repair plans record the search preference as metadata.
Compiler and judge contract v6 preserve alternative branches rather than requiring every
subtype in each example.
Compiler policy v7 also avoids turning illustrative details into extra mandatory
conditions, while preserving every explicitly asserted requirement.

Discovery can work **top-down or bottom-up**, using the same saved family-comparison
workflow and ordinary behavior judge. `family_discovery(action="status")` suggests a
direction and shows current supported detectors available as seeds; these are search
guidance, not evidence of a shared cause or an extra model review.

When at least two current reviewed seeds span distinct failed tasks, stalled confirmed
coverage, repeated top-down edits or the top-down comparison limit schedules one bottom-up
inspection. Before another top-down comparison, new mutating repair plan or round advance,
the agent must record a shared parent or document incompatible causes with checked evidence
using `no_coherent_family(seed_functions=[...])`. Active repairs, evidence gathering and
processing retries remain available. Unchanged seed claims and task support do not repeatedly
trigger the checkpoint across rounds or resume. It uses the existing comparison effort.
Normal completion waits for the inspection and any proposed parent's assessment or explicit
deferral. An exhausted round records a checked proposal for the next ordinary comparison
budget. Explicit time/experiment limits, cancellation and idle-session safeguards still apply.

- Top-down: survey different failed tasks, propose a common mechanism, evaluate a broad
  detector, then add useful specific children. The survey aims for 3/6/8 failed tasks
  for Precision/Balanced/Recall when available.
- Bottom-up: when useful narrow detectors arrive first, retain them and inspect their
  supported cases for a shared causal mechanism. Record a hypothesis with
  `family_discovery(action="record_family", seed_functions=["narrow_a", "narrow_b"], …)`.
  Supply exact source evidence from each current evaluated, causally supported seed,
  plus the common mechanism and boundary/recovery cases. A saved snapshot identifies the
  seed claims. Narrow detectors do not have to wait for an initial family survey.

Submit and independently evaluate the proposed parent, then call `assess_family` with
ordinary judge feedback. For example, substituting canned answers for required computation
and discarding that computation's output may support a computation-bypass detector that
also recognizes other variants. A union of unrelated bugs is coverage, not one generalized
mechanism. Seed purity does not transfer to the parent. Do not set the parent's `explains`
or `depends_on` to its seeds: that would prevent it from finding cases outside them.

The assessment compares observed parent coverage against the seed union, separating
preserved seed matches, unmatched seed cases, and additional failed tasks. Partial records
remain labeled; new matches require their own causal review before claiming broader
support. A parent matching only the seed union has not demonstrated additional recall.
The existing review budget prioritizes newly covered failed tasks, retained seed support
and passing controls when available. Saved review samples report what was actually covered;
the parent's review remains separate from independent confirmation on untouched tasks.
Keep useful seeds while assessing the parent. If no coherent abstraction emerges, retain
their coverage and explore another angle.

Test broad parents against similar traces that share surface cues or unresolved outcomes
but lack the claimed mistaken operation. Pair those contrasts with inspected seed support
and supported new variants across distinct tasks. After a rejected review, use the existing
`smoke_test_feature` controls to test the contradicted condition and recheck those positives.
Repair the causal distinction without adding task-specific exceptions or automatically
reducing the family to its seeds. These controls remain development evidence.

Both directions share the existing 1/3/5 comparison budget for Precision/Balanced/Recall;
there is no additional synthesis quota. Evidence and source task identities are checked
without a separate relationship review or pre-evaluation causal screen. Effort counts
guide search rather than block useful edits. Preserve broad supported parents and inspect
lost support before tightening. Use `explains` / `depends_on` only for intentional execution
dependencies.

No-family findings and deferred hypotheses preserve checked evidence and next steps.
The phase ends after its comparison budget or available failed evidence is exhausted;
it does not require a novelty quota, minimum recall, or invented families. All detector
adherence reviews use 10% of flagged examples, rounded up, with a minimum of three and
capped by available eligible matches. During iterative evaluation, this is based on
observed flagged examples on the scoring panel; final review uses full-dataset matches.
Family comparison does not impose a separate review cap.

`library_coverage` separates matched source tasks from current individually confirmed
causal cases and tasks. Cross-task causal support needs confirmations on at least two
source-verified tasks. Useful one-task findings remain scoped diagnostics, and repeated
rollouts of one task do not establish generalization. The live display reports
comparison effort, failed task coverage, and marginal additional failed tasks/rollouts.
Count overlapping parent/child detections once. Partial observations stay labeled;
exact additional coverage needs complete comparable census records. These signals
allocate search; the three standard ranking objectives are precision, distinct-task recall
and causal purity. Library progress measures the union of confirmed tasks, separately from
each detector's observed task matches.

**Confirmed task coverage** remains a separate breadth measurement.
`pareto_status` and `library_coverage` expose `task_coverage`: the union of current causal
confirmations from detectors meeting the Good precision and adherence thresholds,
and support on two distinct failed tasks. A completed capped sample can supply provisional
search evidence; final qualifying Good detectors still need the full-corpus scan. This count is an
observed lower bound, not an estimate of unreviewed matches or held-out validation.
Repeated rollouts and overlapping detectors count once. A numeric frontier improvement
without broader confirmed coverage does not reset the optional coverage-plateau counter.
The best coverage is preserved across Resume, so dropping and restoring detectors cannot
manufacture progress. Temporary source outages preserve that checkpoint. Evaluations awaiting
causal review remain pending across turns and count once when their reviews finish.

Ordinary discovery completes the planned rounds before the runtime rechecks its minimum
and stopping policy. **Continue until coverage plateaus**, when enabled, additionally waits
for confirmed coverage to plateau and pending work to finish. **Terminate if stalled** uses
its separate validated extra-round counter: useful discoveries, increased Combined Coverage,
and meaningful retained detector improvements can reset it without increasing confirmed task
coverage. Two turns without discovery work trigger recovery guidance, not an automatic pause.
Round boundaries validate retained detectors and allow one bounded refinement exchange;
last-planned and extra rounds use the full corpus. Follow `final_validation_guidance` for
runtime-managed scoring and review, then recheck the actual qualifying minimum. Iterative
edits continue to use the capped sample; do not launch a duplicate final scan.

Use `sample_items(coverage="unconfirmed_tasks")` to search tasks lacking quality-confirmed
coverage. It excludes all failed rollouts of already covered tasks together and preserves
the passing-control pool. Narrow seeds remain available for synthesis; their unconfirmed
tasks remain discoverable. `unmatched` and `unconfirmed` retain their attempt-level meanings.

Use `sample_items(coverage="unmatched")` to investigate diverse missed failures. Check
whether an existing mechanism was missed, evidence/execution is incomplete, or another
family is supported. Smoke-test known positives and contrasts. Missing or abstained
execution is not a confirmed negative. `library_coverage` keeps the selected-corpus
failure denominator fixed and exposes unresolved/error counts separately.
Partial abstentions keep their evidence-inspection advice alongside breadth guidance
from a current comparable panel and reviewed resolved predictions. An all-abstention
evaluation cannot justify broadening a detector.

Ordinary discovery sampling also defaults to distinct source tasks within each outcome
group. A failed and successful rollout of the same prompt can both be inspected to identify
what changed. Set `distinct_tasks=False` deliberately to compare more repeated rollouts;
their task identities and repeat counts remain visible. These contrasts help refine a
causal rule, while confirmed failures on different prompts establish its generality.
Explore similar requirements/actions with opposite outcomes as well as broad samples.
The agent can use `search_items(query=..., is_fail=False/True)` and inspect the evidence;
the sampler does not automatically form similarity pairs. Neither method takes priority.
Benchmark outcomes alone do not label a particular causal mechanism.
After a useful narrow seed, seek that mechanism on another task before cataloguing more
task-specific bugs. A detector can match multiple repeats, but they do not add task breadth.

Iterate with `evaluate_functions(mode="capped")` and the configured `llm_eval_sample`
(default 200 examples); unchanged detector outputs are cached. Detectors with no LLM calls
in their source, helpers or parents scan the full dataset by default. An explicit debug
`limit` still caps both kinds. Cheap repairs derive diagnostic feedback from that same full
scan. Search comparisons remain on the shared panel. Automatic/development LLM repair reviews use
that same panel. These reviews never
silently buy a full-corpus scan. Explicit confirmation may seek fresh evidence outside it;
saved confirmation and processing-retry cohorts remain fixed. Prefer one semantic decision
over overlapping LLM audits of the same evidence unless controls establish a distinct need.
Read sampled labels, sample sizes and intervals; increase the budget deliberately when needed.
Independent confirmation is optional: if seeking fresh evidence of generalization, freeze
the candidate and request eligible reserved confirmation before inspecting final-census
feedback. Supported development reviews can qualify for Good under the shared quality
criteria, with their sample limitations retained. Follow the runtime's final-validation
guidance: run a requested full scan once, or return the summary for driver-managed final
scoring. Errors and abstentions stay unresolved. No new upfront 20% partition is
created, and exposed tasks do not become independent evidence on resume. Full-census
feedback can expose previously unseen cases; tuning on those cases makes later review
of them development evidence. Saving and publishing only preserve existing results.

Interrupting a tool stops new detector/reviewer work and cancels active async judge sessions.
Already-sent synchronous detector HTTP requests drain before the runtime can be reused;
cancelled operations cannot later overwrite library scores or history. Completed checkpoints
remain reusable, and cancellation does not affect other sessions.

For a controlled comparison, use the same dataset, starting library, budget and model
settings for the working baseline and updated workflow. Compare judges on identical frozen
candidates before comparing discovery harnesses with one common judge. Track union failure
coverage, distinct failed tasks, marginal task coverage, false positives, causal confirmation,
unresolved cases and elapsed time. These workflow changes do not by themselves demonstrate a measured
recall improvement.

New/Continue and fresh batch runs use geometric-mean objectives. Recall is the default;
`--optimization balanced|precision|recall` selects a preset for New/Continue, and batch YAML
accepts `config.optimization: recall` (or `balanced`/`precision`). These explicit selectors
override loaded preferences for that new run without rewriting global settings. Existing
custom weights over the three current metrics are retained. The weight editor normalizes
other axis sets before showing sliders for these three metrics. Starting discovery with
older axes (including attempt recall or specificity) selects current Recall instead.
`objective_rank` accepts only `geometric_mean`: settings or a saved run that name another
ranking, such as `weighted_sum` or `product`, fail to load. New runs snapshot their selected
objective settings for Resume.

Loading saved settings and runs preserves their explicit objective fields; omitted settings
fields take the current defaults. Resuming an existing run retains its saved scoring; starting
a new Continue run uses the current objective behavior described above.

For a bounded unattended pilot, use
`autofa new --api-run RUN_ID --optimization balanced --rounds 2 --min-failure-modes 0 --once`.
`--rounds` bypasses the depth picker for New/Continue; disabling the minimum prevents
automatic extra rounds. Resume keeps the original run's objective and discovery state.

Discovery can call `library_coverage` to count distinct failed rollouts with saved matches
separately from rollouts with individually confirmed causal reviews. Parent and child
matches count once. Per-detector counts expose unique contributions and additional cases
beyond their parent. Missing, stale or partial evidence remains explicit; the tool never
extrapolates reviewed confirmation to all matches. `sample_items(coverage="unmatched")`
explores failed rollouts without a current recorded match; `coverage="unconfirmed"`
explores those without a confirmed causal review. Passing controls keep their usual pool.
These filters steer discovery; evidence read through tools is recorded for later
confirmation sampling. Compose a child
using `depends_on=["parent"]` and `explains="parent"`, and distinguish its within-parent
recall from corpus recall.

New and Continue accept an analysis focus without changing discovery orchestration or
detector scoring:

```bash
autofa new --api-run d1e4200611235732bb86815151d13921 \
  --analysis-focus both

autofa continue --library /path/to/library.json \
  --analysis-focus harness
```

`model` focuses on the model's decisions and behavior. `harness` focuses on tool
contracts, environment, orchestration, budgets, and feedback delivery. `both`
asks where improvements can be made: model, harness, both, or insufficient evidence. Selecting a focus
does not enable extra data access or permit grader-label leakage. Claims still
need visible evidence; missing tool observations can limit harness diagnosis.

Taxonomy classification is a separate workflow. Use `autofa classify` for one
classification command, or `autofa run --config FILE` with `workflow: classification`
for a complete selection, multi-judge report and optional publication. Pass its results or a
prepared taxonomy group to
`autofa new --input ...` or `autofa continue --input ...`. See
[Taxonomy classification](taxonomy-classification.md).
`taxonomy_first_pass` selects `llm` or `agentic`; `taxonomy_second_pass` selects
`off`, `llm`, or `agentic`. The `llm` mode uses the HTTP taxonomy judge. The
`agentic` mode uses the separate Judge endpoint harness and model.
`taxonomy_agentic_max_turns` is the positive per-pass interaction budget (128
by default). It counts investigation calls that can reveal new evidence or
taxonomy information for both Claude and Codex.
Citations, review-status updates, and the final submission do not count. Once
the budget is spent, further investigation calls are refused and the judge is
asked to cite what it read and submit or abstain.
`autofa classify --agentic-max-turns none` removes the limit for that command.
Both harnesses also have a 10-minute session timeout. The same bounded pass
setup is available from
**Settings > Taxonomy classification** and the interactive classification flow.

Practical starting points:

- Use **128** for short-to-medium traces with limited tool history and few
  artifacts, such as many GDPVal and SWE-Bench tasks.
- Use **256** for long, tool-heavy traces with extensive verifier output,
  hundreds of events, or several artifacts, as can occur in some
  Terminal-Bench and DeepSWE tasks.
- Use **`none`** for controlled runs over extreme traces that require repeated
  paging through very large logs or artifacts; the 10-minute timeout remains.

If many traces need the last tier, narrower artifact search is a better fix
than setting an ever-higher default.

Persistent defaults in the repository's `autofa.toml`:

```toml
analysis_focus = "both"
skills_dir = "" # Auto-detect the skills folder; or set an explicit path.
skill_paths = []
disabled_skill_paths = []
analysis_harness = "" # Empty = the logs' source harness selects reference packs.
```

CLI flags override these defaults only for that New/Continue run. The run config
stores the selected focus and active skills. Resume reuses that saved
guidance even if defaults change or source files disappear. Use Continue to start
a new run with different guidance.

### Markdown skills

**All Markdown files in the skills folder are active by default, recursively.**
AutoFA injects their complete contents into its discovery agent's context, including
harness descriptions and any taxonomy documents. The agent is instructed to scan
all documents, use the relevant harness/version, and use taxonomy categories as
discovery hypotheses and error-classification guidance. Novel categories remain
allowed. These documents are context, not proof of what happened in an evaluated run.

Harness reference packs are the exception. Markdown under the folder's `harness/<pack>/`
loads only when the pack id matches the analyzed harness, and Markdown directly in
`harness/` (its `README.md`) only with a matched pack. AutoFA compares the source harness
recorded with the logs (each source's, for a mixed run) by whole tokens, split on
non-alphanumerics and ignoring case: `hermes_agent` selects `hermes`, `openai_api_agent`
does not select `pi`, and `lmarena_v3_benchmark_agent` selects none. Set
`analysis_harness` to override the match with one or more pack ids, such as `"codex"` or
`"codex, langchain"`, or to `"none"` to load no pack. Files added through `skill_paths`
always load. The discovery agent, behavior judge, and improvement classifier receive the
same selection, and the launch summary names packs that were not loaded.

In a source checkout the default folder is the repository's `skills/` (not AutoFA's
internal slash-command skills). Installed AutoFA uses `./skills/` if present, otherwise
`~/.autofa/skills/`. Set `skills_dir` or use **Settings > Skills > Select skills folder**
to choose another location. **Settings > Skills** lists active files and lets you
exclude individual documents, re-enable all Markdown, or add files from elsewhere.
Changes save automatically. New Markdown files become active in the next New/Continue
run without an enable action. Files are never rewritten by AutoFA.

At launch AutoFA lists loaded documents, user exclusions, and files skipped because
they are not Markdown. Full contents, source paths, SHA-256 hashes, and this inventory
are saved in the run config. Resume uses that snapshot, even after source
edits or deletion. A resumed run also keeps its pack selection: runs created before
pack matching still send every pack they saved, so their judge reviews stay reusable.
Use Continue to start a run with the current selection. Unreadable, empty, non-UTF-8,
explicitly selected non-Markdown, or unscannable inputs fail explicitly. Limits are
64,000 bytes per file and 256,000 bytes combined; AutoFA rejects oversize selections
instead of silently omitting content. Symlinked directories must be selected as the
root or copied into it; they are not silently skipped or recursively followed into cycles.

### Improvement targets

After behavioral judging, the improvement classifier, an auxiliary LLM call, assesses
where each failure mode could be improved: the model, the harness, both, or neither
when the evidence is insufficient. It uses the configured `judge_model` and optional
`judge_base_url` / `judge_api_key_env` / `judge_api` overrides, up to six
judge-confirmed trace excerpts, and the selected skills. Each assessment names the
concrete change for each target, the supporting item IDs, and a proposed regression test.

The assessment steers discovery rather than the results you browse. The discovery agent
receives it with each behavior-judge result. When the analysis focus is `model` or
`harness` and the evidence supports only the other target, the agent is asked to find a
concrete change within the selected focus or to pivot. The assessment is saved with the
detector in the library and in the report's embedded data, but the HTML report does not
display it. A missing credential or provider error records the assessment as
`unavailable`; it never stops discovery or changes detector scores.

These are proposed interventions, not demonstrated fixes. A model-originated
mistake can still be harness-addressable, for example through clearer tool errors.
Changing a detector's source or claim marks its previous assessment stale.

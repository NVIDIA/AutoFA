# Configure models and effort before analysis

Before the first launch, help the user choose the model and reasoning settings for each
relevant role. Inspect the effective configuration and available authenticated harness/provider
capabilities first. Present a compact recommendation with the current or proposed model,
effort/thinking setting, endpoint/authentication route and quality/cost/latency rationale.
Reuse explicit preferences, including a request to use defaults; ask only about meaningful
unresolved tradeoffs. A report-only request does not need model setup.

In the same setup phase, follow [cloud reuse](cloud-reuse.md) when the workflow can consume
existing detectors. Offer relevant libraries and resolve the seed choice before launch;
model setup can proceed independently while the user decides.

For discovery, the default configuration permits the escalation sequence Luna baseline →
Luna medium → Sol medium → Sol high. Resolve concrete deployment IDs from the user's endpoint;
the default IDs are preferences, not evidence of model availability. Explain both the API
route and reasoning setting: current API `auto` does not mean automatic reasoning effort, and
fixed-label routes may ignore medium/high reasoning. Confirm the ladder, comparison allowance
and spending scope with the user, or reuse an explicitly saved agreement. If the user declines
escalation, set `classifier_profiles = []`; disabled escalation means no alternative-model calls.

Keep this policy in `model_configuration`, snapshot resolved profiles into native runs, and
offer `driver.py preferences --path FILE --settings-file FILE --reason TEXT` to save the user's
agreed choices for future campaigns. Resume keeps its saved choices. Native AutoFA compares
profiles on retained controls and adopts eligible results; the driver verifies that the allowed
comparison was performed. A failed intermediate tier must not be adopted merely to reach the
next tier: use native tested-tier progression and preserve unsuccessful evidence.

## Resolve where the settings came from

Verify the executable's source checkout and the selected configuration file separately.
A fresh download does not isolate settings: this version can import `~/.autofa/config.toml`
on first launch, and a driver-created campaign file can itself be a copy of that old profile.
Inspect explicit environment overrides, project/ancestor configuration and native first-launch
inheritance before initializing a new profile. Show the selected path and whether its values
come from current defaults, an existing profile or a copied profile. Record the source path
and fingerprint when known; a destination filename alone does not establish provenance.

When the user requests current defaults, create a new campaign-scoped configuration through
the matching version's native initializer, explicitly binding its new path before creation.
In this version, an explicit `AUTOFA_CONFIG` pointing to a nonexistent campaign file bypasses
implicit home-profile import and ancestor lookup. Do not copy an entire old profile and then
change only the minimum detector count or objective: other historical settings would remain.
Carry over only agreed connection/model settings and credential references, naming those
overrides. Preserve existing files and explicit reuse choices; Resume keeps its saved settings.
If a user explicitly selects an existing profile, describe its inherited settings rather than
claiming they are the downloaded version's defaults. Starting without imported detectors is a
separate choice from using fresh configuration defaults.

## Select the discovery objective

For failure discovery, also resolve the optimization objective before launch. A request
for defaults on a new analysis uses the installed new-analysis default (currently Recall:
50% failure-task recall, 30% detector adherence, 20% normalized precision). An inherited
Balanced profile is an existing configuration choice, not the new default. Preserve explicit
user-selected objectives and saved objectives on Resume; do not rewrite global configuration.
For a new run, pin the agreed preset through the discovered native interface. Current
interfaces are `--optimization recall` for New/Continue and `config.optimization: recall`
in batch YAML; use the user's chosen preset instead when specified.
For explicit custom weights, leave the preset selector unset and pin the selected named
axes and weights (`objective_rank` stays `geometric_mean`) in a separate campaign
configuration. Inspect the native schema first, and preserve unrelated settings.
Show the effective preset and named percentage weights with the model choices, record them
in campaign `optimization`, and check the saved run's `objective_settings` after launch.
Do not describe inherited equal thirds as Recall or silently migrate a resumed analysis.

## Separate the roles

Discover the installed version's role definitions, model catalogs, settings schema and
precedence through its help, configuration docs and native settings code. The following are
current configuration landmarks, not a fixed model list or a substitute for runtime discovery:

| Role | What to help configure | Current native landmarks |
|---|---|---|
| AutoFA harness | Discovery harness, model, supported reasoning effort/thinking and any applicable thinking budget | `harness`, `model`, `effort`, `thinking`, `thinking_budget_tokens` |
| Behavior judge | Independent review harness, model and supported reasoning controls | `behavior_judge_harness`, `behavior_judge_model`, `behavior_judge_effort`, `behavior_judge_thinking`, `behavior_judge_thinking_budget_tokens` |
| LLM detectors | Provider/deployment, API route, credential environment-variable name, reasoning/request options and permitted alternative classifier profiles | `feature_model`, `feature_api`, `feature_base_url`, `feature_api_key_env`, inherited hub settings, `feature_request_options`, `classifier_profiles` |

These are models executing AutoFA; the model whose failures are being analyzed is identified
separately by the source cohort.

The behavior judge's `auto` harness follows the AutoFA harness; it still has its own model
and effort. In this version, `judge_model` and other `judge_*` API settings configure auxiliary
HTTP classifiers, including improvement suggestions and taxonomy; they are **not** the
behavior judge's model. Explain this distinction when interpreting existing configuration.
For other workflows, discover their actual roles and controls rather than configuring unused
discovery settings. Taxonomy, preflight and future workflows can have different API defaults.

## Make a useful recommendation

- Start from the user's task, data modality, deadline, expected volume and allowed spending.
  Explain which role needs stronger reasoning or image/trace understanding and which repeated
  calls dominate runtime/cost. Offer a concise choice when the tradeoff matters; do not make
  the user fill out a configuration questionnaire or silently choose maximum effort everywhere.
- Inspect native token/cost reporting for the installed harness and API routes before giving
  a spending estimate. Separate provider-reported USD, price-table estimates and account credits;
  a subscription's reported estimate is not an incremental invoice. Resolve permitted model
  pricing by exact provider/deployment using the native per-run pricing interface when needed.
  Reuse supplied rates and record their source/date; do not invent prices or substitute a similar
  model. Unknown prices or unavailable usage remain unknown. Tracking does not itself enforce
  a hard spending cap, so explain the available enforcement boundary for an explicit budget.
- Query the installed harness's model metadata and supported effort levels; use the provider's
  available deployments/capabilities for API calls. Do not hard-code historical model names,
  assume every model supports the same effort controls, or infer API access from CLI login.
  Preserve custom deployment IDs. Mark unknown access, vision support and cost as unverified.
- Distinguish per-call reasoning effort from output-token limits, discovery rounds, evaluation
  samples, review budgets and worker concurrency. More workers do not make a classifier reason
  better; higher effort can consume more tokens and time without guaranteeing better recall.
- Use provider-supported request options for detector reasoning. An explicit `{}` means
  provider defaults, not reasoning off. Option tables can replace defaults rather than merge;
  preserve intended options when editing. Fixed-label Decisions may ignore generation options:
  verify the effective API route before promising that an effort setting takes effect.
- Configure allowed classifier alternatives when useful and within scope, so AutoFA can compare
  them natively. Keep the behavior judge and auxiliary classifier routing explicit when changing
  detector profiles. Compare on retained controls and evaluate adopted settings through native scoring.

## Apply and verify

For managed discovery, inspect the actual launch with
`autofa config inspect --run-config /path/to/run.yaml --json`. This makes no inference calls
and does not create/migrate settings. It reports the configuration origin, resolved role models,
named objective weights, permitted profiles, limits, publication choice and monetary readiness.
If first-launch harness detection is still pending, complete normal setup before agreeing a
snapshot; an inspector cannot certify login or model access.

These setup contracts apply to discovery. For `workflow: classification` configurations,
use the native `autofa run --config /path/to/classification.yaml` workflow and inspect its
taxonomy model/pass settings directly. Use `autofa new --config` or `autofa continue --config`
with that classification file for a later taxonomy-to-discovery handoff; do not pass it to
discovery `config inspect` or `config prepare`.

Save the agreed JSON, then use
`autofa config prepare --run-config /path/to/run.yaml --expected /path/to/agreed.json --output /path/to/new-launch-dir`.
The output contains a private scoped profile, batch YAML, setup contract and managed launch
specification with file hashes. The agreement also pins the native source and interpreter.
Native execution checks the snapshot before readiness checks and again
after run creation. Use the returned environment and arguments; do not reconstruct them from an
older home profile. Set the delivery choice through the driver before inspecting the batch.
Add `--require-capability vision` to inspection when image capability is explicitly required;
that requirement is included in the agreed fingerprint. Do not infer capability from a dataset
name. Resume retains native live settings and adoptions.

Quality escalation and availability recovery differ. When the baseline is returning transport
errors, use native `classifier_profiles` recovery with the approved profiles; its availability
path can test a fallback without a successful baseline. Valid negative control results use the
normal comparison path. Preserve source-supported positive/negative controls and native review.
When no authorized model works, propose a concrete configuration/wait/partial-delivery choice.
Do not treat unavailable experiments as scientific stagnation or impose an unexplained ban on
model changes that prevents the agreed goal.

When dollar estimates are requested, treat `pricing.needs_pricing_setup` and
`missing_rate_roles` as setup work. Reuse supplied verified rates; otherwise retrieve current
official provider/model pricing before launch. Save the URL, retrieval/effective date, units,
processing/context/cache assumptions and exact deployment binding in the scoped `usage_pricing`
snapshot, then re-inspect it before preparing the launch. Cover discovery, behavior judging,
LLM detectors and permitted alternatives. Do not leave every role unpriced merely because the
harness exposes credits. Fetching public pricing does not require an inference probe.

An OpenAI-compatible protocol does not identify the billing supplier. Internal gateways and
subscriptions may need a user/organization rate card; a documented public list-price equivalent
can still support a useful reference estimate. Preserve custom deployment IDs and state the
mapping assumption. Do not substitute a similarly named model or silently select a context or
speed tier. When no defensible mapping/rate exists, explain exactly which part remains unpriced;
missing billing data alone does not require blocking a run without a spending constraint.

Native harness readiness `token_reference_estimate_available` means observed tokens can be
priced separately; it does not promise a dollar counter or convert credits. Read
`token_cost_estimate` in native cost/status results. If an active run lacks rates, use
`autofa supervise pricing --run RUN_ID --settings-file rates.json --estimate-only`, then reread
cost. This updates only the reference estimate and needs no analysis restart. Keep it distinct
from `monetary_view`, provider measurements and the spending guard. Report the discovery forecast
and finalization/delivery reserve separately, refreshing both after meaningful model/work changes.

Use the matching version's native settings/configuration interfaces. Prefer a campaign-scoped
configuration when supported and identify whether a change affects this launch, a saved run
or the shared profile. Verify configuration precedence and accepted fields: a batch run spec
need not expose every field in the full configuration. Do not invent batch keys or flags.
Use a verified scoped config-file mechanism for settings absent from that spec.

For Resume/Continue, inspect saved harness settings, active fallback records and adopted
classifier profiles before choosing overrides. Resume can restore saved discovery/judge
choices over the active TOML file. Changing a project/global file alone is not proof that an
existing run will use it. Use supported update/adoption paths and checkpoint when needed;
do not rewrite saved identity or make old scores appear current under a different model.

Check credentials by presence/reference without displaying their values or placing them in
campaign records. Prefer read-only metadata/readiness checks. Distinguish a dry preview from
a write or a paid probe; perform bounded probes only within the user's authorized scope.
Inspect the current `preflight classifiers` help: preview/readiness,
`--write-config` and `--probe` have different effects. A text probe does not establish vision
or reasoning quality.

Before launch, show the effective role/model/effort choices and any inherited, unsupported or
unverified settings, together with the discovery objective. Record selected settings, scope,
configuration fingerprints and allowed alternatives in campaign `model_configuration`,
without secrets. After native launch, verify
the run's saved effective choices and record fallbacks or overrides instead of claiming that
the requested settings necessarily took effect. Follow [supervision.md](supervision.md) for
later controlled changes.

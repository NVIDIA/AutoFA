# Evidence, finalization and delivery

## Explain findings and useful next actions

After a report is generated, help the user understand the main findings, not only the scores
or report link. Start from the actual report and its native evidence/validation receipts;
do not reconstruct findings from detector names or a progress recap. For an existing-report
request, summarize the available artifacts directly without starting another analysis.

Explain the most consequential supported mechanisms, affected sources/tasks, overlap and
remaining uncovered errors. Include a few representative report/evidence links when useful,
the requested coverage/count outcomes, and limitations that change interpretation. Separate
observed behavior from causal explanations and proposed fixes. Scale the summary to the
workflow: taxonomy findings, efficiency bottlenecks or other outputs need their own native
units and evidence, not forced GOOD-detector counts.

Recommend a small prioritized set of useful next actions, covering improvements to the
analyzed model/harness/system as well as further AutoFA investigation when warranted. Tie each
action to a finding or unresolved question, specify the source/subset and proposed native
workflow, explain expected value and tradeoffs, and give a bounded budget and success/stop
criterion when known. Mark unknown cost or benefit as unknown. Examples to adapt to evidence:

| Evidence in the results | Useful follow-up |
|---|---|
| Recent rounds add validated marginal coverage and promising mechanisms remain | Continue from the current library with bounded additional rounds and a marginal-coverage or deadline stop condition. |
| Misses concentrate in a source or error cluster | Run targeted analysis of that explicitly selected subset, preserving source-qualified IDs and membership. |
| A broad pattern has an unresolved explanation, contradictory review or missing artifact | Deep-dive on representative traces/images and boundary cases through native investigation; state the question to resolve. |
| Detector execution fails fixed controls or lacks needed image/reasoning capability | Compare permitted model/effort profiles or diagnose evidence delivery before spending more discovery rounds. |
| Scores/reviews are unfinished or stale | Complete native validation first so further choices use current evidence. |
| A supported mechanism suggests a concrete product change | Propose a model/prompt/tool/harness or data/evaluation improvement and a regression experiment to test it; keep causal benefit a hypothesis until measured. |
| Coverage has plateaued or delivery constraints dominate | Recommend stopping exploration, acting on findings or checking generalization on fresh data rather than adding loops without a new hypothesis. |

For a “failures only” follow-up, distinguish a focused discovery subset from its validation
cohort. Preserve the original report and full-cohort denominators. A failures-only cohort
cannot establish false-positive rate or precision on passing rollouts. Retain suitable
passing/boundary controls and plan native validation on the original or a fresh representative
cohort; label subset recall separately and do not present it as improved full-dataset coverage.
Do not discard earlier exposure history when creating the subset or continuing its library.

Record the proposed follow-ups with report/version references and separate recommended,
authorized, running and completed states. Suggestions alone do not launch paid work or reopen
a stopped campaign. Carry out follow-ups already authorized within the remaining goal/budget;
otherwise present them for the user's next instruction. Preserve the completed report as the
baseline and revise the goal when a continuation is actually chosen.

## Interpret native results

Use native metrics, receipts and exact source/context identities. Distinguish per-detector
recall, deduplicated failed-attempt union and distinct-task coverage. Preserve the original
denominator; count each failure once; include per-source coverage for mixed data. Qualify
sampled results, unresolved predictions, stale contexts and lower bounds. Discovery GOOD can
be provisional; final counts require current full-corpus measurement and native representative review.

Inspect finalization selection. Disclosed gaps in excluded candidates differ from missing
evidence for retained GOOD/OK. Neither process exit nor a saved report proves completion.
Use structured native contracts/receipts, not one console message. Finished discovery,
achieved detector count, completed validation and successful publication are separate facts.

For source-audit discrepancies, preserve the native verdict; record exact claim/context/item
and contradictory or ambiguous evidence. Show qualification and union sensitivity, including
coverage excluding the disputed detector when useful; state assumptions/bounds. Another
review cohort does not automatically resolve a known contradiction. Use native reconciliation
without manual score replacement or altered thresholds. Register audit exposure before reuse.
Benchmark PASS/FAIL labels alone do not establish a judging error. The native judge assesses
task requirements, the claimed behavior and relevant outcome/recovery from source evidence;
the original benchmark labels remain authoritative for the reported precision/coverage
denominators. A confirmation on a PASS-labeled item can be a benchmark/source disagreement.
Identify the actual source contradiction before claiming a policy violation or proposing a
different verdict. Do not treat exclusion of an entire detector based only on such labels
as corrected coverage; a sensitivity calculation remains hypothetical.

## Finish or publish saved work

For a generic stop request, follow [graceful stopping](operations.md#graceful-stopping):
ask about final full-dataset scoring and report publication, recommending yes to both.
Normal stopping aims to deliver a report; the answers determine whether it is fully validated
or saved/partial, and cloud or local. Honor explicit immediate-stop and no-upload choices.

Reserve finalization/transfer time. Reuse successful predictions/reviews; avoid duplicate full
scans. Use native validation-only actions when needed, preserving discovery rounds and sources.
Report unmet goals honestly.

Keep a pinned deliverable checkpoint during improvement: native run/source/library identity,
validation scope, report path/hash and the coverage/count values it supports. If newest work
is unfinished at the delivery boundary, select the agreed coherent checkpoint or explicitly
partial snapshot, rather than pairing an older report with newer scores. Preserve the live
work and disclose what the fallback excludes. A native finalize acknowledgment is not a
rendered/published report; finish the complete delivery chain and verify the immutable version.

For discovery, read the final native usage snapshot after scoring, judging and authorized
highlighting have settled. Include accounted tokens, reported versus estimated USD, missing accounting and
cost outside discovery rounds; do not treat the last discovery round as the final total.
Prefer complete monetary accounting when available. If billing is incomplete but
`token_cost_estimate.available` is true, also give its `estimated_cost_usd` as an observed-token
reference (API-equivalent when public rates were used), with the source assumptions, unpriced
observations and missing-token gaps. Do not add that independent total to reported dollars or
credits. An incomplete $0 subtotal is not free work; do not report only “unknown” when a sourced
token estimate is available. Accounting observations are not necessarily individual API calls.
Usage lives in operational records, separate from the HTML report. State the accounting
observation time and report costs in the conversational delivery summary. Do not add cost,
audit, authorship or hypothetical-sensitivity sections to the report. Preserve audit records
and explain relevant qualifications separately. External driver costs remain separately
attributed or unknown. Analysis cost and the analyzed model's efficiency cost measure different work.

For “enough exploration; just upload,” persist stop and verify owned work/watchers stopped.
Inspect the whole call chain: ordinary CLI publication can prepare optional highlights.
For a saved-results-only discovery upload, use `autofa publish --run RUN --saved-only --json`.
It takes the native run lock, preserves saved scores and reviews, skips scoring/judging/highlights,
and reads back the immutable publication. `--verify-only` performs reconciliation/readback
without an append. Other workflows retain their existing native publication paths. Preserve
saved grades and incomplete-result qualifications.

## Publish and verify

Use native publication, retaining requested service/library/branch/run and source identity.
Publication can append an immutable version; retain its exact ID and URL. A moving “latest”
link is not a verified immutable artifact.

Measure library JSON, report HTML and assets separately against real client/server limits.
Use deterministic lossless native packing/restoration of evidence/history. Preserve definitions,
helpers, metrics, current verdicts, full evidence and provenance; never truncate reviews to fit.
A client-only limit increase cannot fix server rejection. Do not change shared infrastructure
as an automatic workaround.

Inspect capability/version compatibility. A narrowly verified unsupported-field fallback
requires complete retained source identity/semantics; never delete arbitrary metadata on HTTP
422. Older readers must not treat unsupported packed evidence as complete. Restore and verify
before Continue.

Record the destination, source identities, exact version URL and artifact hashes in the
outer campaign record. Native report/library preparation may normalize bytes; distinguish
that transformation from changes to evidence. Use the complete native runtime so publication
preserves persistence history.

After a timeout or uncertain reply, inspect remote versions and source/hash identity before
retrying. Publication uses a direct append API with no automatic request-key reconciliation;
a retry can create another version. Download the exact version and verify hashes, evidence
restoration, displayed metrics, report/source links and validation scope. `driver.py verify`
checks local bytes only; it cannot certify remote identity or metric correctness.

## Deterministic discovery delivery contract

Agree `goal.delivery.destination` (`"cloud"` or explicit `"local"`) at setup. Cloud means
publication to the configured cloud service; preserve an explicit local-only choice. Set
`publish_on_stop` when delivery from a stopped campaign is chosen; it permits delivery to the
selected destination, including local rendering. Do not silently set it false and thereby
drop an agreed report. A generic stop invokes the two graceful-finish questions; an unanswered
default does not authorize final inference or cloud upload. Reuse explicit answers in the
stop request or a standing stop policy. Keep `summary_required: true` so delivery includes
the findings in this conversation. Ordinary successful completion still uses the agreed
automatic publication choice without another confirmation.

Build the native specification with the contract before inspecting/freezing setup:

```sh
python3 agent-skills/autofa-driver/scripts/driver.py delivery prepare-run \
  --campaign /path/to/campaign --run-spec /path/to/input-spec.json \
  --output /path/to/native-run.json
autofa config inspect --run-config /path/to/native-run.json --json
```

The input and generated specification are JSON objects, accepted by the native YAML reader.
The helper maps cloud to `execution.publish: true` and local to `false`, even if an earlier
input carried the opposite default. Preserve the other execution fields. Agree the resolved
setup, then use native `config prepare` and the managed launch/wait protocol. Generic native
batch defaults are unchanged.

Bind the existing native executable/config and resulting run IDs through the supervisor
configuration, or `driver.py delivery configure --campaign DIR --native binding.json --run RUN`.
Managed runs retain their own frozen launch binding; it takes precedence over mutable global
delivery settings. The native publisher checks the saved setup fingerprint and agreed service
URL/mode/credential reference before any remote request, while permitting native code upgrades.
The binding contains `argv` (an array), optional `cwd` and `env`; explicit `config`/`home` map to
`AUTOFA_CONFIG`/`AUTOFA_HOME`. Store paths and credential references, not secrets. Shared
foreground and detached coordination use:

```sh
python3 agent-skills/autofa-driver/scripts/driver.py delivery reconcile \
  --campaign /path/to/campaign --run RUN --allow-publish
python3 agent-skills/autofa-driver/scripts/driver.py delivery summary \
  --campaign /path/to/campaign --summary /path/to/findings-message.json
python3 agent-skills/autofa-driver/scripts/driver.py delivery check --campaign /path/to/campaign
```

Reconciliation first reads the current immutable version. After a lost reply it checks the
saved prepared snapshot against visible versions before any retry; unknown server outcomes
stay `publication_uncertain`, with a usable local report. Confirmed local preparation failures
remain retryable. The API remains direct append: do not promise exactly-once delivery or
blindly retry an uncertain append. Verification restores packed evidence and compares source
run/branch/model, the full normalized library, report metrics and source identities.

The summary input contains `outcome` (`complete` or `partial`), `text`, supported `findings`,
`next_actions` and a `cost` object (or `status: unknown` with a reason). Include every exact
verified report URL in `text`, the count/combined-recall results with denominators and scope,
target shortfalls, material blockers and a useful next action. The helper pins this message
to verified report metrics and artifact hashes and returns `sha256`. Send that exact message
in the host conversation, then acknowledge the observation with `supervisor.py ack
--summary-sha256 HASH` and the existing host/session/generation identifiers. A detached adapter
must acknowledge delivery of `final-summary-HASH` to the agreed channel. A queued file is not
delivery. If monitoring was paused or the host closed, native publication may already be
finished. Reattach, reconcile the saved delivery, and send the verified findings/link; do not
rerun analysis or label an unsent summary delivered.

`driver.py complete` and supervisor completion both require fresh native readback, terminal
native validation/outcome and that exact summary acknowledgment. A completed-with-gaps or
infrastructure-blocked native result can be delivered as **partial**. When explicitly agreed, a
user-stopped run with saved evidence can also be delivered as partial without paid finalization;
the saved stop remains in effect. Repeating completion/check against the same accepted contract
reuses its original summary acknowledgment; reopening or changing the goal/artifacts requires
new verification and delivery. Successful upload never
turns it into a quality success. Source preparation failure without saved results cannot pass
this completion gate. Cloud outage keeps delivery pending/blocked and does not justify more
paid review turns.

## Attribution and recap

Separate native discovery, driver authorship/imports/repairs, native optimization and native
scoring/review. Preserve origin author/source/library/version and later edits, inspections,
controls/exposures. A native score proves neither development independence nor infallible judging.

Quantify assistance from receipts: interventions, externally inspected samples, introduced or
modified candidates, subsequent native rounds/experiments, runtime patches and measured resource
use. Unknown cost/authorship remains unknown; do not invent percentages. Disclose attempted or
rejected fixes and material limitations.

Deliver a compact table: source/run, completed rounds/workflow progress, validated versus
provisional GOOD/OK, requested coverage with denominator/scope, final state and exact report
version link. State target shortfalls, disputes and meaningful driver contributions. Include
final accounted cost and any omissions; projections remain estimates, never a final bill. Record
delivery independently from optimization success. Close owned background work and verify no
loop can revive a stopped/completed campaign.

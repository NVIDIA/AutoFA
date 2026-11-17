# AutoFA usage and cost accounting

AutoFA records resources consumed by failure discovery and its validation,
finalization and explicit classifier-setup probes, including available harness and HTTP-model
usage. This is separate from [efficiency metrics](efficiency.md), which
measure the original model runs, and from the driver's own host-agent usage.
Taxonomy classification, standalone efficiency metrics and the cheating preflight check retain their
existing accounting and report formats; this feature does not add cost cards or ledgers to them.

Inspect saved accounting without inference or a live model session:

```sh
uv run autofa supervise cost --run RUN_ID
uv run autofa supervise status --run RUN_ID
```

The `status` response includes the same `usage` summary. It contains token counts, accumulated
USD and credits, groups by round/role/model, projection details and accounting notices.
Rows outside discovery rounds retain their costs, including finalization when recorded.
The ledger counts usage observations; a cumulative harness observation can include several
provider requests. It is not a request counter or a billing invoice.

## What the numbers mean

- **Reported USD** is monetary usage reported by a provider interface as native usage.
- **Estimated USD** includes harness estimates and estimates from explicitly configured API
  rates. Claude SDK dollar totals are estimates; a subscription estimate is not necessarily
  an additional charge. Codex monetary/credit availability depends on the installed server
  and account; unsupported or absent fields stay unknown.
- **Known cost subtotal** adds the known reported and estimated USD. If some observations have
  no cost data, this is a partial subtotal, not the full spend. Credits are shown separately;
  AutoFA does not convert credits, currencies or subscription allowances into invented dollars.
- **Observed-token reference estimate** (`token_cost_estimate`) prices measured harness and API
  tokens using the supplied rates, including the measured cached-input discount. A public-rate
  reference is an API-equivalent estimate, not the subscription charge or gateway invoice.
  This independent view can include work that already has reported dollars; never add its total
  to the known-cost subtotal. It does not change spend admission or resolve missing billing data.
- **Tokens** retain available input, output, cache and reasoning fields. Categories can overlap
  and differ by provider. Missing fields are unknown; do not add every category to the total.
- **Missing usage/cost** counts observations without the respective measurement. A request that
  failed without returning usage may still have incurred cost. Cache hits do not imply a new
  provider call; unobserved work from old runs cannot be reconstructed as zero-cost work.

Codex compaction notifications do not expose separate token counts, and its ordinary thread
token snapshots can omit that work. AutoFA records observed automatic and explicit compactions
as deduplicated accounting gaps, preserving the reported token counters and any independently
reported thread cost or credits. The summary exposes `compaction_usage_gaps` and a notice;
tokens remain a partial measurement. AutoFA does not invent the missing counts or recover them
retroactively from older runs. A known USD subtotal of zero with missing cost observations
means cost is unavailable, not that the work was free.
An available token-reference subtotal can still give a useful dollar estimate in that case.
Its `complete` flag describes coverage of the observed token data only; missing compaction or
other token measurements keep it partial. It never certifies that an invoice is complete.

Native adapters record available usage and cumulative-session deltas in the durable run
ledger. Resume keeps the ledger; cumulative snapshots are not repeatedly added as fresh spend.
Within a native operation, accounting reuses a bounded set of connections, commits each
event immediately, and closes the connections when the operation ends. Live reads use SQLite
locking and retry one database read error. An unwritable directory can use an immutable
snapshot only when there is no pending WAL; otherwise missing accounting is reported explicitly.
Continue creates a new run with its own accounting. Sum distinct run identities when reporting
a campaign. For a known local source with a spending policy, Continue transfers its remaining
allowance under the source analysis run's lock, carries prior spend and already-consumed overtime, and
prevents the source from spending the same grant again. Continue from the newest child if
that allowance was already transferred. An active source or unresolved source spending must
be resolved first; a new run is not a new spending grant. Changed global defaults do not
enlarge an inherited grant. Remote or imported libraries do not supply spending history:
the driver must supply an explicit new allocation or an independently reconciled campaign
baseline for those campaigns. Outer driver sessions and separately launched tools without a
native usage scope are not automatically included; report their cost separately or as unknown.

Round attribution follows the measurements that the harness exposes. Codex snapshots usage
before a native round transition as well as at turn completion. Claude's authoritative usage
arrives with its final SDK result; an interrupted turn without that result remains unknown.
If one Claude turn spans several discovery rounds, AutoFA retains the measured turn total
outside the individual round buckets instead of fabricating a split. Those gaps can prevent
a per-round monetary forecast, but do not make its measured dollar total unknown or block
spend admission. Codex compaction token gaps likewise remain separate from a complete native
dollar counter. Missing tokens do not make independent monetary measurements unavailable.

Explicit classifier probes accept `preflight classifiers --probe NAME --usage-dir PATH`. Without
that option, their accounting directory is a sibling of the configuration file named
`<config-stem>.probe.usage`; the result prints `usage_dir` and `autofa_usage`. Inspect the
current help for other probe options. Read-only classifier inspection does not run probes
or create paid work. This optional setup command configures discovery detectors; it is not
a dataset preflight and is not automatically invoked by discovery. The ordinary discovery
startup endpoint-readiness probe remains separate.
Inspect a probe's accounting with `autofa supervise cost --usage-dir /path/to/config.probe.usage`.

## API pricing

When a provider returns tokens but no monetary total, an explicit per-run rate card can produce
an estimate. Use exact provider/model identifiers from the ledger and rates appropriate to the
endpoint/contract; there is no bundled assumption that a similarly named deployment has the
same price. For example, create `rates.json` with your verified rates:

```json
{
  "version": 1,
  "currency": "USD",
  "rates": [{
    "provider": "openai_compatible",
    "model": "exact-deployment",
    "endpoint": "https://provider.example/v1",
    "input_per_million": 1.0,
    "cached_input_per_million": 0.2,
    "cache_creation_per_million": 1.25,
    "output_per_million": 4.0,
    "source": "user-supplied rate card",
    "effective_at": "2026-11-03"
  }]
}
```

These numbers illustrate the format; they are not a model price recommendation. `endpoint`
is optional. Cache rates are optional: the supplied input rate applies to all input unless a
distinct cache rate is declared. A distinct cache rate requires the corresponding measured
token count; if that count is missing, the estimate is unknown. Configure an inactive
saved run through the native lock-protected interface:

```sh
uv run autofa supervise pricing --run RUN_ID --settings-file rates.json
```

For an explicit classifier probe, substitute `--usage-dir EXISTING_DIRECTORY` for `--run RUN_ID`.
Settings are written atomically under the corresponding lock. When `--usage-dir` targets a
saved run directory, its usual native run lock applies.

Rates apply to future observations; recorded costs retain their provenance and rate hash.
Provider monetary data takes precedence. Configuration does not rewrite historical spend,
change model routing or enforce a hard budget. The driver should review accounting support
and pricing alongside initial model/effort choices and preserve explicit spending constraints.

New discovery runs can carry an agreed `usage_pricing` snapshot in their configuration.
Native discovery startup installs it before the first discovery/readiness call.
Resume preserves the
run's existing rate file rather than replacing it with changed global defaults. Explicit
classifier probes also install configured rates in their standalone usage directory before
dispatch. Probe spend is separate from the later discovery run and belongs in the campaign's
setup allowance.

When no applicable configured rate or provider amount exists, the driver can research current
official pricing during setup. It must resolve the exact deployed model and endpoint; the
ledger provider `openai_compatible` describes a protocol, not a billing supplier. Researched
rate rows use `basis: "public"` or `basis: "proxy"`, an exact API `endpoint`, a source HTTP(S) URL
and `retrieved_at`. Native `codex` and `claude` harness rows can omit the endpoint because their
token streams do not identify an HTTP billing endpoint; retain the exact harness/model identity.
Optional `billing_provider` and `pricing_model` retain the public SKU
mapping. Proxy rates also require `assumptions`, such as a public list-price equivalent for
an internal gateway. This remains an estimate, not a claim about the gateway's contract or
a subscription invoice. No current public prices are bundled, and installing the program
does not authorize an alternative-model call or paid pricing probe.

Cost status also exposes a separate `derived_estimate` for saved unpriced API observations
when applicable rates become available later. It retains the rate snapshot/hash and the
assumed historical pricing date. It never changes `usage.sqlite3`, provider charges or the
original `totals`. `monetary_view` combines the recorded subtotal and this derived estimate
and reports the remaining unpriced observations. Separate harness cost counters are excluded
from historical token repricing to avoid double charging; unpriced credits remain unknown
dollars. Missing discounted-cache counters also keep an estimate unavailable rather than
assuming a zero cache count.

The separate `token_cost_estimate` also includes harness `usage` observations, which Codex
records independently of its dollar/credit counters. Cumulative baselines and cost-only rows
are excluded; saved incremental token measurements are priced once. Its `groups` and `by_round`
retain model/role/round attribution, rates and assumptions. Unknown models, missing rates,
missing cache counts and unmeasured work appear as explicit reasons rather than zero prices.
Reported dollar counters arriving later do not get added to this reference estimate.
Routine status retains the reference totals, reasons, round totals and pricing digest, with
group counts and a `details_command`. Full model/role groups and repeated rate provenance stay
in `supervise cost` (or explicit status `--full-snapshots`) so long runs keep monitoring compact.

To obtain a reference estimate for an active run without changing its ledger, billing estimates
or budget checks, install a separate rate card:

```sh
uv run autofa supervise pricing --run RUN_ID --settings-file rates.json --estimate-only
uv run autofa supervise cost --run RUN_ID
```

This atomically saves `usage_estimate_pricing.json` without taking over the native analysis.
The reference estimator uses that file when present, otherwise the run's `usage_pricing.json`.
The CLI reads saved token data and performs no model calls. A running analysis does not need to
restart to inspect the new estimate. Use the ordinary pricing command on an inactive run when
the intended change is to the rates used for future recorded API costs and spend admission.

Fetch exact-model rates from the provider's current official documentation, such as
[OpenAI API pricing](https://developers.openai.com/api/docs/pricing), and retain the retrieval
date, processing/context assumptions and any gateway mapping. If the mapping or required rate
cannot be supported, leave that part unpriced and state the specific missing information.
Public list prices are not a conversion from Codex credits into dollars.

Some SDKs retry requests internally without exposing usage for every attempt. The ledger
records the measurements visible to AutoFA and flags this accounting limit; a successful
final response is not evidence that hidden retry costs were zero.

## Projection and final reports

The native forecast uses completed rounds with sufficient cost accounting and requires at
least two eligible rounds. Its response states the measured basis, completed/remaining rounds
and reason when unavailable. It is an estimate using past work, not a guarantee: model changes,
new cohort sizes or expensive validation can change future cost. Future finalization is excluded
from the estimate; already accounted finalization remains in the accumulated total.

For an agreed discovery `campaign_policy`, the forecast instead uses a compatible observed
discovery/validation burn rate and the remaining authorized time, stopping before the delivery
reserve. The basis includes ordinary validation work and the union of measured paid-dispatch
intervals; idle gaps and paused/offline time between calls are excluded. Older records without
dispatch durations receive an unavailable forecast rather than a misleadingly low burn rate.
It does not treat the normal round count as a terminal denominator. The result separates
remaining discovery spend from `finalization_delivery_allowance_usd`; that allowance is an
agreed reserve, not a measured prediction. At least two compatible completed rounds, a
positive observed time interval and sufficient accounting are required. Model/rate/scope
changes, unknown costs and missing round/duration attribution can make the forecast unavailable.
Changing only the requested detector count does not invalidate a comparable cost basis.
Spend-only campaigns without a deadline do not invent a time/workload forecast. Consumed
overtime uses its saved effective boundaries and is never granted again by a cost read.

Spending enforcement is an **estimated guard**, not a provider billing hard cap. Concurrent
in-flight work, hidden retries, delayed usage and unobservable outer-driver charges can exceed
it. Multi-run campaigns must allocate explicit per-run shares plus any separate driver/setup
allowance so the shares do not each claim the whole campaign budget. Refresh final recorded
totals after validation, highlighting and other chargeable finishing work settles; show
unknown amounts and unmet delivery constraints explicitly.

Paid dispatches reserve estimates for their actual role, provider, model and endpoint. A
small detector LLM endpoint call never reserves a whole discovery turn's cost. Committed incremental
SQLite totals and a transaction revision keep admission reads bounded without relying on
filesystem timestamp resolution. Reservation files are durable before dispatch; a killed
owner cannot restore its allowance by restarting. Active and orphaned estimates remain
separate from recorded charges in status.

`unknown_cost_policy="stop"` blocks new paid work with unresolved interrupted charges.
`"allow_estimate"` can count a retained, priced reservation against the remaining allowance;
it does not treat an unpriced lost call as free. Both policies accept complete configured
price estimates and provider-reported money. True unresolved monetary gaps also block new
paid finalization; a finalize request cannot bypass that limit. Saved results can still be
delivered with their validation gaps, without new inference.

After obtaining a provider receipt, reconcile an interrupted physical dispatch using its
pending reservation ID and the **total** charge for that dispatch:

```sh
uv run autofa supervise reconcile-cost --run RUN_ID --id RESERVATION_ID \
  --cost-usd 1.25 --source "Provider billing receipt reference"
```

The run must be inactive. Previously measured amounts stay intact; the missing delta and
source are recorded with a stable identity, and identical reconciliation is idempotent.
This resolves monetary accounting without inventing token counts or detector evidence.
The example amount is a placeholder: use the actual supported charge, never a guessed receipt.

The driver reports complete monetary totals when available. When billing is incomplete, it
also reports an available `token_cost_estimate.estimated_cost_usd` as a separate observed-token
reference, with its priced/unpriced observations and missing-token qualifications. Do not stop
at a $0 subtotal or “unknown” when a sourced token estimate is available, and do not call the
number of accounting observations a count of provider requests. Present projected totals
separately; a token reference does not make an unavailable billing forecast trustworthy.
Check spending after meaningful progress rather than launching
model calls to fill missing billing data. In-flight work may incur cost after a cooperative
stop request, so tracking alone is not a hard spending cap.

Usage remains in the operational ledger and the read-only `supervise cost` / `supervise status`
views. Failure reports retain their existing layout and do not embed analysis-cost cards or
usage snapshots. Publication freshness depends on the saved library and source identity,
not on changes to usage accounting. Report final accounted cost separately in the delivery
summary, distinguishing reported costs, estimates and missing accounting.

Discovery's `config inspect` also reports monetary readiness for the configured harness,
behavior judge, detector baseline, permitted fallback profiles and auxiliary classifier.
`estimates_ready`, `needs_pricing_setup` and `missing_rate_roles` summarize rate coverage and
unresolved setup work, with an actionable `next_action`. Matching harness
rates report `token_reference_estimate_available`; billing remains unverified until runtime.
This is a rate-coverage check, not a claim that provider accounting has been verified. Required
batch readiness runs after native accounting initialization; standalone classifier probes keep
their own usage directory and rate snapshot. Failed calls that return no counters remain unknown
even after token prices are supplied. Include measurable outer-host costs separately and label
unobserved supervision spend explicitly. Forecasts require enough compatible observations;
until then, report the known subtotal and the native reason that projection is unavailable.

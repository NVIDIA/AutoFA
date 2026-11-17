# Failure library and saved results

[Back to README](../README.md) · [Local traces](local-traces.md) · [Detector reference](detectors.md)

[Local Library server](#local-library-server) · [Data cards and branches](#data-cards-and-branches) · [Where things live](#where-things-live)

## Local Library server

The local dashboard is included in the normal AutoFA installation.
Launch from the checkout with `uv run --python 3.12 autofa`.

Launching `autofa` starts the local Library Server automatically, including when
cloud publication is configured. The home screen shows a clickable
**Library** link and an **Explore failure library** menu
option. The default address is <http://127.0.0.1:8765>. AutoFA stops the server it
owns when you exit, after final publication. Saved data persists across launches.

The browser includes the local libraries in your active AutoFA profile, reports,
trace evidence, and immutable published versions and branches.
Publication accepts `regex`, `llm`, and `python` detector kinds in version-1 and
version-2 libraries, preserving uploaded detector source and kind.
Libraries saved with pre-GA strategy kinds (`hybrid`, `deterministic`, `trace`,
`composed`, `cross_field`) as a detector's top-level kind are rejected on load,
download and publication; recreate or re-export those detectors with the current
AutoFA. Local saves that use them are not indexed (a previously indexed snapshot
stays available).
Libraries sort by original publication time, falling back to save time when publication metadata is absent.
Saved catalogs, runs, and large report assets are indexed in the background without
modifying their files or making browser requests wait. The dashboard shows progress
and updates as libraries become available; enriched reports are added as new
versions. Subsequent launches reuse the index and refresh changed saves. Exiting
AutoFA stops indexing before shutting down its server.

The local browser and publication backend are configured independently. To change
the local browser port, add this optional setting to your active `autofa.toml`:

```toml
local_library_base_url = "http://127.0.0.1:8766"
```

The source distribution supports cloud publication and browsing. When signed in,
automatic cloud publication sends a lightweight report with saved failure cards
and passage quotes. It links to EvalRDL traces only when a valid upstream reference
is available; local-only examples have no cloud trace link. Full traces stay
in the local snapshot and are not uploaded as report assets. The Local Library
loads those traces and their exact passage highlights on demand.

To publish to the local service instead, replace the existing publication URL
with its address:

```toml
remote_library_base_url = "http://127.0.0.1:8765"
```

If `local_library_base_url` is unset or empty, an existing loopback
`remote_library_base_url` also selects the local browser address; otherwise the
browser uses port 8765.

For a dashboard that runs independently:

```sh
autofa serve
```

AutoFA reuses an existing server for the same profile and leaves it running on
exit. If several sessions reuse an automatically started server, it closes when
the owning session exits; use `autofa serve` for an independent shared lifetime.
`autofa serve --port 8766` changes the standalone port; set the local browser URL
to that same address for automatic reuse.

The server binds to loopback by default and stores its database, artifacts, and local
token under `$AUTOFA_HOME/server/`, or `~/.autofa/server/` by default.
Use the same `AUTOFA_HOME` for the CLI and standalone server; set it explicitly
when you want a separate research profile. The local server uses the profile
token and does not require cloud sign-in or use cloud credentials.

To host the library on a separate machine, run
`autofa serve --host 0.0.0.0 --port 8765` there. In each publishing session, open
**Settings → AutoFA Cloud / library server**, select **AutoFA server**, and enter its address.
Set `AUTOFA_LIBRARY_TOKEN` to the receiving server's token. AutoFA transfers the
complete managed trace snapshots, including full text and captured artifacts,
then publishes the report. Transfers stream from disk and reuse existing snapshots;
the receiving viewer works after the publishing machine disconnects. Remote browser
access also requires the token. See [setup and configuration](local-traces.md).

Changing the publication backend does not change dataset access or model-provider
settings. Start a new run when switching publication services: publication IDs
belong to the service that created them. AutoFA rejects a change of publication
service when it conflicts with the run's recorded service.

## Data cards and branches

Cloud and local Hub cards are grouped by source data within an owner's namespace.
An EvalRDL source is identified by its service origin and run ID. Local inputs use
stable content hashes, so moving a file does not create a different dataset.
Analyses using the same set of input datasets share a card, even when their model
or benchmark labels differ. Each saved version retains its model, benchmarks,
source identity, and report; dashboard labels describe the latest version.

Fresh analyses create separate branches within that card. Re-publishing or
continuing an analysis keeps its branch and appends an immutable version. The
source-data label on each card is separate from **Latest analysis**, which names
the AutoFA execution. Search accepts source IDs and full local hashes.

Owners can open **Merge with another data card** on a card's detail page to combine
two cards explicitly. Select the destination card and confirm the merge. Branches,
versions, report history, and existing links remain available. Future analyses of
either dataset join the merged card, which is labelled **Explicitly merged**.
If either cloud card is private, the merged card stays private.
Historical card IDs remain readable. Deletion requires the current card ID;
requests through old merged-card aliases are rejected before any data is deleted.

Analyses without usable source provenance keep a card of their own instead of being
grouped by model and benchmark. Cards upgraded from pre-release databases may show
**Historical mixed sources**; their history is retained.

Hub databases from pre-release versions that were never upgraded to schema 7 are
not opened; startup stops without changing them. To start a new database, stop the
Hub and move `autofa-library.db` and its `-wal`/`-shm` files out of
`$AUTOFA_HOME/server/` (`~/.autofa/server/` by default).

## Where things live

### Recorded contributions and source audits

AutoFA retains candidate authorship separately from scoring. New native submissions record
their origin; later native edits preserve a driver-authored or imported origin. Existing
libraries with no authorship record remain **unknown**. Native evaluation does not establish
native authorship or independent confirmation.

The native `library_import_candidates` tool previews imports by default. Applying imports
unscored candidates with pinned dependencies/helpers and preserved exposure history. A
different existing definition or a claim already owned by another detector requires the
ordinary repair workflow. Fresh evaluation is required; source scores and match records do
not become destination grades. The generic `library_load` restore/merge behavior is unchanged.

`record_external_inspection` reserves exact destination item/task identities as development
exposure and saves actor/source receipts. Unknown or ambiguous item mappings are rejected.
`record_source_audit` records original-artifact findings against a detector and its saved
review identity; it never rewrites verdicts. Optional `proposed_verdict` values support
explicitly hypothetical adherence sensitivity on that exact frozen cohort. Changing cohorts
does not resolve an earlier dispute. `audit_sensitivity` also computes the native union with
specified disputed detectors excluded, keeping the failure denominator unchanged.

These receipts travel in `library.json` through Continue, export and publication. Reports
disclose recorded assistance and unresolved audits, separating hypothetical sensitivity from
native scores. Inspect the installed tool schemas before use:

```sh
autofa supervise tools --name library_import_candidates
autofa supervise tools --name record_source_audit
```

For an inactive saved run, `autofa supervise invoke --run <run-id> --tool <tool-name>
--arguments-file <json-file>` uses the native run lock and tool admission. Calls can mutate
saved results; choose the preview or read-only action when inspecting.

### Saved files and diagnostics

Action failures show their cause, failed setup stage, and a next step before
returning to a menu. Interactive errors wait for acknowledgement, and the last
outcome remains visible after the menu redraw. Explicit cancellation and Ctrl+C
are reported separately. One-shot (`--once`) failures exit with a nonzero status.
The same handling covers discovery, continuation, resume, settings, library
browsing, publication, and agent-session errors.

`~/.autofa/cli-errors.log` retains exception chains. `cli-actions.jsonl` records
action starts, setup stages, API/run identity, outcomes, and the actual installed
source/Python paths, including failures before an analysis run exists. If a
process is killed, its last recorded stage identifies where it stopped.

Remote source imports explicitly select the unnamed conversation lane (`lane=`).
Grader/verdict lanes and graders' model calls remain excluded from model evidence
([datasets](datasets.md)). Model-call evidence is
recovered separately from the original raw record, including recorded later
requests, rather than combining auxiliary lanes into a misleading event sequence.
Nullable per-event tool-call counts are reconciled against the complete,
validated rollout-level call list, and Responses API `output` reasoning is
retained as assistant reasoning. Source cache versions track normalization changes;
new imports rebuild prepared datasets while reusing validated downloaded records.
Existing runs retain their bound dataset.

```
<repo>/
  autofa.toml                   # active project settings (git-ignored)
  skills/                       # Markdown context, recursively active by default

~/.autofa/
  config.toml                   # fallback settings; explicit profiles use this name
  .env                         # optional keys
  sources/<store_id>/            # immutable NeMo Gym / Harbor imports
    manifest.json
    tables/                     # normalized EvalRDL tables
    dataset.parquet             # analysis-ready rows
  server/                       # Local Library database, artifacts, and profile token
  converted/<hash>/dataset.parquet   # generic JSONL conversion cache
  converted/remote/<origin>/<run_id>/ # API origin + canonical run ID cache
    current.json                      # latest completed export generation
    exports/<generation>/
      dataset.parquet
      source.json
  cache/remote-libraries/<hash>/     # service URL + immutable version ID cache
    library.json
    metadata.json                    # concrete logical/branch/version selection
  library/{model}/{benchmark}/
    library.json                 # durable catalog
    library.provenance.json      # local run + cloud branch/version lineage
    failure_cards.html           # offline Model Failure Cards (click link in CLI)
  runs/{run_id}/
    config.json                # dataset binding, benchmarks, top_x, …
    library.json               # working detectors and helpers for this run
    failure_cards.html         # regenerated on /dashboard
    scores_history.json        # per-eval snapshots (+ frontier_value, n_functions)
    session.json
    CLAUDE.md / .claude/       # Claude workspace when that harness is selected
    AGENTS.md / .agents/      # Codex workspace when that harness is selected
```

Libraries auto-save on session exit and on `/save`. Choose **Explore failure library**
from the home menu to browse saved libraries and reports. For terminal browsing,
run `autofa library` —
each row is a **model · discovery-benchmark** catalog (not model-only); press **`p`** to
publish the selected local library immediately or **`d`** to delete it (with confirm). Opening a
catalog prints a clickable `file://` link to **Model Failure Cards** HTML (mechanism,
metrics, expandable detector buckets/source). Each failure card has a stable
detector-name link that can be copied from the cloud dashboard.

Reports start light and remember your theme choice. Click a heading to collapse its card.
The failure overview shows three links; use **Show more** to expand the rest.
Reports display **Good** detectors first, then **OK**, with higher **Coverage** first
within each group. Lower-grade and unreviewed candidates remain in the saved library
and report data, but do not appear as report cards or count toward its failure-mode total.
The CLI also displays **Weak**, **No matches**, and **Not evaluated** after those groups.
Normalized precision, detector adherence, then name break coverage ties. This display
order is independent of discovery optimization weights. The live CLI shows the first
five in that order.

When discovery has recorded connections between current visible detectors, the report
adds a **Connected failures** overview and matching labels on their cards. Each group
lists its members in card order (A.1, A.2, …), and each card names its typed synthesis
or explanatory links; a connection does not imply a subtype. Cards retain their quality
and coverage order. Missing, stale, removed or hidden findings create no empty groups.
Grouping uses saved links only, makes no model calls, and does not change discovery,
grades, coverage, minimum-detector targets or stopping decisions.

Scores appear below the title: **Verdict**, **Coverage**, **Normalized precision**, and
**Detector adherence** (behavior purity). **Failure rate when flagged** appears in the
detector details. Hover or focus each information icon for an
explanation. A brief AI review summarizes the saved judgment. Open **Code and examples**
and a bucket marked **↗**; use **Open viewer** for its source trace when available.
Detailed review data remains in the download. Report viewers use the page's
full width and size the iframe to the viewport height. The Local Library displays
saved reports with the current UI, preserving their stored scores and examples.

With a cloud library configured, the first interactive `autofa` or `autofa home`
launch offers GitLab sign-in. This one-time prompt never runs in CI,
with `--once`, or from operational subcommands. The home header then shows
the current cloud identity (or a compact, nonfatal offline/expired status);
local AutoFA remains available regardless of cloud status.

You can also manage sign-in explicitly:

```bash
autofa login
autofa publish --run <run-id>
autofa continue --remote-library <remote-library-id>
# Select a specific branch and version:
autofa continue --remote-library <id> --remote-branch <id> --remote-version <id>
autofa logout
```

For cloud mode, `autofa login` opens a same-origin browser authorization flow and saves the
resulting `autofa_cli_…` service token in the OS keychain, scoped to the
normalized `remote_library_base_url`. The token is validated against `/api/v1/me`
before it is saved. No plaintext fallback is used when the machine has no usable
keychain backend.

For CI and existing PAT-based workflows, set the configured environment variable
(`AUTOFA_LIBRARY_TOKEN` by default) instead:

```bash
export AUTOFA_LIBRARY_TOKEN="<your-service-token>"
```

An explicit environment credential always takes precedence over the keychain.
`autofa logout` best-effort revokes and removes only the saved AutoFA CLI token;
it cannot unset or revoke a GitLab PAT supplied by the shell.

When the configured token is present, AutoFA automatically publishes the final
library and failure cards when a native session exits. Cloud reports
link Data API examples to their exact source rollouts instead of uploading
trace copies. Examples without a valid upstream source reference retain
their failure cards and passage quotes, with full traces available locally.
The Local Library loads managed traces from their snapshots; standalone HTML/ZIP
exports retain full evidence. Publish to a configured AutoFA server to transfer
managed local snapshots to another machine; see [local traces](local-traces.md).
New cloud pages are public by default. The bundled library server offers no
visibility controls; a source-data card or owner profile that is already private
(for example, set by an earlier server or kept when two cards merge) stays
owner-only. Each immutable version records both AutoFA's execution harness and
the harness that produced the analyzed logs. Data API
sources retain exact run references: cloud report pages link to the evaluation run,
and examples inside failure-card inspectors link to their exact Data API
rollouts. Continue offers to reuse those evaluation runs by default; Resume
redownloads them only when its bound cache is missing. Versions also record
whether they were published by the CLI, Browse menu, or native session completion.
Cloud libraries are versioned: browse and Continue select a logical library, then
a branch (skipped only when there is exactly one), then an immutable version,
newest first. Fresh runs use their local run ID as a stable branch key, so
re-publishing appends another version to that branch. A run continued from cloud
keeps the selected logical library, branch, branch key, and base version, while
the new version records the new local run ID as its `source_run_id`.

The adjacent `library.provenance.json` sidecar gives local Continue and `p`-publish
the source analysis run and cloud lineage without adding machine-specific metadata to the
portable `library.json`. A catalog copied without its sidecar starts a fresh lineage;
publish it with `autofa publish --run <id>`. When a library exceeds the upload
limit, bucket membership IDs and, if necessary, review history and verbose evidence
are losslessly packed into the version-2 envelope. Loading restores every record,
including localisations and prior exposure. Packing is deterministic, checksummed
and bounded to 128 MiB of total decompressed evidence. Detector sources, helpers,
active verdicts and aggregate scores stay readable; the local working library keeps
its ordinary format. Smaller libraries use the existing representation. Schema errors
never silently drop source metadata. Library, HTML and asset upload limits are checked
separately.

Publication uses the direct version-append API. It does not create a local publication
intent, negotiate request-key reconciliation, or republish because usage accounting changed.
After an uncertain network result, inspect the remote versions before retrying: a retry can
append another version. Source identity and saved assessments are preserved.

Publication attempts are
recorded in the run's `publication.json`; failures also go to `cli-errors.log`,
remain visible in the menu, and show an `autofa publish --run ...` retry command.
They never discard the local library.

Credentials are sent only to the configured AutoFA library service. They are never
written to `config.toml`, run configuration, provenance, cache metadata, logs, or
uploaded artifacts.

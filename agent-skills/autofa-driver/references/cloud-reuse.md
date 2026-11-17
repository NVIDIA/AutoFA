# Find and offer existing cloud detectors

Before launching a workflow that can consume detectors, check the configured AutoFA Cloud
for relevant libraries and ask whether the user wants to reuse them. This can save discovery
work; AutoFA still performs target-cohort evaluation, optimization and review. Honor an
explicit fresh/unassisted analysis or an already selected seed without repeating the question.
Report-only and workflows without a detector input do not need this setup step.

## Discover and compare

Use [workflow discovery](workflow-discovery.md) to identify the installed version's cloud
browse/list, branch/version, download, Continue and candidate-import interfaces. Use the
configured cloud service and existing access; distinguish it from a local catalog or loopback
server. Reading cloud metadata does not require uploading the user's data, launching inference,
publishing a library or importing detectors into a run. Keep authentication values out of output.

Start with metadata and follow pagination. Compare candidate source identities, then inspect
relevant branch/version records: a logical library's latest-version summary does not describe
all of its history. Prefer exact dataset/cohort matches, but offer useful related-dataset
detectors with the difference stated. Do not automatically choose the newest global branch.

| Evidence | Interpretation |
|---|---|
| Same canonical Data API origin/run IDs or native local content identity | Same source data; still compare model, benchmark and selection filters. |
| Same data with different model, subset, split, modality or source harness | Related cohort; explain what needs fresh compatibility/evidence checks. |
| Same benchmark name, title or similar failure description | A possible transfer candidate, not proof of an exact dataset match. |
| Missing or conflicting provenance | Match is unverified; do not invent an identity. |

Use native canonicalization and dataset identity metadata when available. File basenames,
cache paths and AutoFA execution `source_run_id` are not dataset identity. In this version,
remote data runs are identified through `reference_data`; its run IDs differ from the AutoFA
analysis run ID. For a combined analysis, compare the whole deduplicated source set. A library
for one constituent source is a partial match, not an exact match for the combined input.

If no relevant detectors are found, say which accessible sources were checked and proceed
with the authorized fresh analysis. Missing configuration, authentication failure, offline
service, an incomplete scan or unavailable metadata means the check is incomplete, not that
no detectors exist. Offer the appropriate native sign-in/configuration or a fresh analysis;
do not make cloud access a prerequisite for local work. Reuse a recent check when inputs and
service access are unchanged instead of asking on every status poll.

## Present the choice before import

Show a short list with the report link, dataset/cohort match and differences, analyzed model,
immutable library/branch/version, date, and detector names/count when available. Inspect saved
reports or a verified local download when metadata lacks details. Saved GOOD/OK counts,
recall and coverage describe the source cohort; label missing results as unknown and do not
promise the same scores on the user's dataset. Include required helper/dependency imports
and any destination name conflicts in the proposal.

Ask one concise question, for example: “I found these detectors for this dataset. Import the
selected detectors, start fresh, or inspect them first?” Identify the exact recommended
selection; offer whole-library reuse or a supported subset as appropriate. Wait for the
choice before applying candidates or starting work whose seed depends on it. Independent
model/configuration checks can continue. An earlier explicit selection already supplies the
decision; a timeout or lack of reply does not. Record a decline so the driver does not keep
offering the same candidates unless the user changes scope or asks to reconsider.

## Apply the selected version natively

Pin the version and expected content hash that the user reviewed. A cloud download/cache is
not an import into an analysis. After the decision, use native Continue or candidate transfer
as appropriate; preserve definitions, dependency closure, helpers, lineage and prior exposure.
Follow [supervision](supervision.md) for ownership/checkpoints, native import previews, conflicts
and unscored target candidates. Never replace an existing detector silently or transplant
source grades, predictions or validation receipts as target-cohort results.

Preserve the user's input runs and cohort explicitly. Continue can offer to reuse the seed
library's original data and publication branch; importing detectors does not authorize switching
to those inputs or appending results to that cloud lineage. Verify the resulting native run's
source binding and destination before analysis/publication. Keep publication within the
campaign's existing authorization. If the installed workflow cannot seed the requested data
or select the requested detectors, explain that concrete limitation and use a supported path.

Record the lookup service/time/scope, match evidence, proposed and selected detector names,
library/branch/version/hash, user decision and native receipt in campaign sources/goal or
intervention records. Record destination mappings, dependencies and conflicts actually resolved.
Have AutoFA evaluate and review the imported candidates on the target cohort, then optimize
or discover additional mechanisms as requested. Attribute reused detectors separately from
new native discoveries in progress updates and the final findings.

## Current interface landmarks

These are verified landmarks in this checkout; inspect the installed contract before use:

- `autofa.data.remote_library_api.RemoteLibraryClient` exposes paginated `list_libraries()`,
  `list_branches(library_id)`, `list_versions(library_id, branch_id)` and `get_version(...)`.
  The current list method has no dataset search argument; filter metadata through the public
  client instead of inventing a `search` command or flag.
- `resolve_library_ref(...)` accepts explicit branch/version selectors;
  `materialize_remote_library(...)` caches a selected immutable version and provenance,
  checking its hash. Omitted selectors can resolve to a different latest version later.
- `autofa continue --remote-library ID --remote-branch ID --remote-version ID` is a launch
  operation, not a read-only lookup. Verify its dataset selection and continuation semantics.
- Discover `library_import_candidates` through `autofa supervise tools --name
  library_import_candidates`. Its `names` selection includes required dependencies;
  `apply=False` previews and `apply=True` imports unscored candidates. It needs a native target
  runtime; do not initialize an analysis merely to list cloud libraries.

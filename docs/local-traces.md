# Importing local runs and opening their traces

Choose **Local eval logs (NeMo Gym / Harbor)** in dataset selection, then select
the NeMo Gym run folder or standalone Harbor job. Keep the run's configuration,
rollout logs, and companion files together. AutoFA recognizes declared reader
layouts before treating their nested files as independent datasets. Generic
Parquet and trajectory JSONL inputs use the generic file loader.
Automatic detection uses the bundled public reader dispatch. A standalone Harbor
job needs its own configuration, including `jobs_dir`; an unrelated `config.json`
is insufficient. A Gym or other supported host that embeds Harbor is imported as
the host's complete run. NeMo Gym model-call captures may be symbolic links to a
shared store outside the run folder. The import follows such a link when the file
it points to is named `*.capture.jsonl`; it does not check that the capture belongs
to this run. It keeps the request parameters and tool names of the captured calls
that the run's records name. Other links that lead outside the run folder are
refused, and links to folders are not followed. Import only run folders, and linked
captures, that you trust.

For recognized runs, the bundled readers and importer produce versioned schema
tables. The bundled AutoFA exporter supplies the canonical columns and labels;
the same rich normalization used for downloaded runs adds the full conversation,
tool arguments/results, model requests, and available execution evidence.
Grader records remain outside detector evidence. The tables do not say which NeMo
Gym model calls a grader made, so import version 4 reads each rollout's raw record
and withholds those calls' events and request messages ([datasets](datasets.md)).

When a conversation begins with an assistant greeting, the importer retains the
exporter's system text and first user message unless a complete agent input is
recorded. Requests from auxiliary lanes, including a user simulator, remain trace
evidence and do not replace the analyzed agent's input. The updated Gym reader
keeps tau2 simulator instructions out of the agent conversation.

The resulting immutable snapshot lives beneath `$AUTOFA_HOME/sources/<store_id>`
(`~/.autofa/sources` with the default source profile). It includes normalized tables,
analysis parquet, conversion diagnostics and captured local artifacts. Cache
identity includes full input-file digests, library and adapter revisions, and
explicit metadata overrides. Moving an unchanged input folder does not create a
new snapshot. Changed logs create a new snapshot; existing reports retain theirs.
Import and adapter versions are part of that identity, so selecting raw folders
after an importer update can create a new snapshot. Existing snapshots and their
saved reports remain immutable. To refresh an analysis, start a new import from
its original raw folder.
Reopening a managed dataset restores a missing `source.json` reference from the
validated manifest. Existing references must agree with that manifest. The import
API's `force=True` verifies a fresh conversion without replacing its saved snapshot;
it does not repair damaged tables. Restore a damaged snapshot from a backup, or
re-import the original logs in a different `AUTOFA_HOME` profile.

Reader findings are retained in `diagnostics.json`. A run containing quarantined
rollouts or ambiguous multi-agent task/repeat identities is refused with an
explanation. The upstream importer enforces one rollout per task and repeat within
a run; the agent name labels that rollout. Benchmark names retain the harness's spelling. Missing model or
benchmark metadata remains `unknown` unless explicitly supplied. An agent's model
reference is not inferred to be the model under test: it may name a router.
Names come from the schema's run record, with AutoFA's optional
`eval_metadata.json` fallback (`model_id` or `model`, and `benchmark` or `dataset`).
The local selection wizard offers optional **Tested model** and **Benchmark**
fields when those names are missing. Leave a field blank to retain `unknown`, or
press Escape to return. Supplied labels create a separate immutable import; known
names and existing Parquet selections are left as recorded. Supplying labels
currently rereads the original logs once. The Python API also supports overrides:

```python
from autofa.data.local_import import import_local_run

source = import_local_run(
    "/path/to/harbor-job",
    model_id="my-model",
    benchmark="my-benchmark",
)
print(source.parquet_path)
```

Recognized runs use the exporter's outcome semantics: reward at least 0.5 and a
status other than `failed`.

In the Local Library, **Go to passage** keeps the exact detector-field highlight.
Reports backed by managed local imports keep a small index of examples and load
each full trace from its verified local snapshot when opened. Indexing or
publishing the local report does not copy the entire corpus into report assets.
**Open viewer** opens the bundled trace viewer on the current Local Library
server at the original rollout, event, part, and lane, with the passage text passed
to its native highlighter. **Show more** loads the full
selected part inline, beyond the preview limit. Failed loads can be retried; loaded
text remains available when collapsed and expanded again. **Full text of selected
event/part** also opens a separate text page. The full-screen
report retains the same local link handling. No separate trace server is needed.

A viewer link can name text to highlight: `#/rollout/<run>/<task>/<repeat>/e12?lane=&text=…`,
in URL text-fragment syntax, or a browser's own "Copy link to highlight" (`:~:text=`).
The highlighter can load text beyond collapsed previews. Text absent from the linked
event is sought in the rest of the rollout. Viewer highlights use text matching;
**Go to passage** retains the report's exact detector-field character offsets.
Report links include surrounding context when needed to distinguish repeated text.
If that context cannot uniquely identify the selected occurrence, the link opens
the source event or part without selecting a different occurrence; the exact
highlight remains available in the report.
The Library, its reports and the viewer share one **Dark**/**Light** choice, set
from the top bar.

Local links survive server restarts and port changes because reports store a
relative viewer address and a snapshot identity. Keep the source snapshot while
retaining its reports. Copying a report file alone does not transfer the trace;
publish to another AutoFA server to transfer its local data as well. Standalone
reports and offline exports include full evidence; the
dashboard loads evidence on demand.
Original input logs can be removed after import, but only artifact bytes actually
captured in the snapshot are available for later downloads. External artifact
URLs are not fetched during import or viewing.

To keep the library on another machine, run the standalone server there:

```sh
autofa serve --host 0.0.0.0 --port 8765
```

The server prints the path to its token file. It can run without a configured
discovery harness. On each publishing machine, set `AUTOFA_LIBRARY_TOKEN` in the
shell or AutoFA profile `.env` to the contents of that file. Open **Settings →
AutoFA Cloud / library server**, choose **AutoFA server**, and enter its HTTP(S) address, such as
`http://analysis-server:8765`. The same form is available in the discovery wizard.
The equivalent configuration is:

```toml
remote_library_mode = "server"
remote_library_base_url = "http://analysis-server:8765"
remote_library_token_env = "AUTOFA_LIBRARY_TOKEN"
```

The server accepts authenticated publications with the report and its full local
trace data. Its bundled viewer reads the transferred data on that machine, so the
publishing session need not stay running. Browser access uses the same server
token. Your publishing machine's Local Library remains available independently.
After configuration, new sessions publish automatically; `autofa login` checks
the connection and `autofa publish --run <run-id>` publishes a saved run. A run
already published to a different service retains its original publication lineage;
start a new run when changing services.

Trace transfers have separate 64 GiB upload and unpacked limits per snapshot,
configurable on the receiving machine through `AUTOFA_SERVER_MAX_TRACE_BYTES` and
`AUTOFA_SERVER_MAX_TRACE_UNPACKED_BYTES`. They do not use the 64 MiB report-asset
limit. The receiver checks the complete inventory and file hashes before making
a snapshot available. Retries reuse an identical installed snapshot. Ordinary
parquet inputs retain full report sidecars and the existing report-asset limits;
the native viewer uses the managed snapshots created when importing raw logs.

Cloud publication (`remote_library_mode = "cloud"`) keeps its lightweight reports
and uses available EvalRDL links; it does not upload the local trace corpus.

Initial reading and conversion still hold the run in memory. Subsequent rich
trace hydration uses batches bounded by rollout count and captured text size;
one larger rollout is processed whole. Select the original harness folder or a
managed `dataset.parquet`, rather than individual normalized schema tables.

The [component notice](../third_party/evalrdl/README.md) lists the copied library
components, the bundled revision, the upstream tests that accompany them and their
current redistribution status. This integration does not install or depend on the
hosted evaluation service.

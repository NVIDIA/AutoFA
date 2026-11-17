# Hermes — Trace / log format

**At a glance:** Hermes emits several **distinct** JSONL trace channels, not one: training **trajectories** (ShareGPT `conversations`), **batch** outputs, an uploadable Claude-Code-style **session trace** (mandatory secret redaction), opt-in **MoA traces**, plus a rotating `agent.log` and structured stream diagnostics. Knowing which channel a file came from is essential to reading it correctly.

## How it works

- **Training trajectories** (`agent/trajectory.py`): JSONL, one object per line `{"conversations": <ShareGPT list>, "timestamp": ISO, "model": str, "completed": bool}`; file `trajectory_samples.jsonl` (completed) or `failed_trajectories.jsonl`.
- **Batch runs** (`batch_runner.py`): per-batch `batch_{n}.jsonl` with `{"conversations": trajectory, ...}`; collects tool/reasoning stats; CLI `python batch_runner.py --dataset_file=… --run_name=… [--resume]`.
- **Session trace upload** (`agent/trace_upload.py`): `build_trace_jsonl()` emits **Claude-Code-style JSONL** — per-line entries carrying `sessionId`, `model`, content **blocks**, and tool-calls-as-blocks; uploaded via `upload_session_trace` to a **private HF dataset** `{user}/hermes-traces`, path `sessions/{session_id}.jsonl`. **Secret redaction is mandatory** — `_redact(force=True)`; if redaction fails it raises `TraceRedactionError` and refuses upload. CLI: `hermes trace upload` / `/upload-trace`.
- **MoA traces** (opt-in `moa.save_traces`): `<hermes_home>/moa-traces/<session_id>.jsonl`; each record captures the model/provider/temperature for each reference agent and the aggregator output. Best-effort; never breaks a turn.
- **Main log file:** `agent.log` (rotating ~5 MiB). Records are tagged with `session_id` for `hermes logs`.
- **Usage insights** (`agent/insights.py`): `InsightsEngine` computes token/cost/tool-usage reports for `hermes insights` / `/insights`.

## Where errors can originate (for the trace *reader*)

- **Channel confusion:** a `*.jsonl` file could be a trajectory (`conversations`), a session trace (Claude-Code blocks), a MoA trace, or a batch file — each has a *different schema*. Identify the channel before parsing.
- **Refused upload on redaction failure:** a session trace won't upload if `_redact` throws or if redacted tool args aren't valid JSON — a run may have no uploaded trace at all.
- **Missing HF token** → upload aborts with guidance.
- **Trajectory write failure is swallowed** with a WARNING (non-fatal) — a "missing" trajectory may just mean the write failed.

## Sources

- `nousresearch/hermes-agent` `agent/trace_upload.py` — session-trace JSONL, mandatory redaction, HF upload.
- `nousresearch/hermes-agent` `agent/trajectory.py`, `batch_runner.py` — trajectory and batch JSONL schemas.

> Note: `agent/trajectory.py` is thin (75 lines); the real trajectory conversion lives in `run_agent.py::AIAgent._convert_to_trajectory_format` (not deep-read).

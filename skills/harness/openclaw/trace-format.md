# Openclaw — Trace / log format

**At a glance:** Logs are **JSONL** (one JSON object per line), daily-rolling at `/tmp/openclaw/openclaw-YYYY-MM-DD.log`, rotating at 100 MB into `.1`–`.5`. Machine-filterable top-level fields include `agent_id`, `session_id`, `channel`. **Transcript text, audio, and provider item ids are never copied into log records.** Session transcripts are persisted separately as JSONL (`sessions/*.jsonl`).

## How it works

**Logs** (`/gateway/logging`, repo `docs/logging.md`):
- **Format:** JSONL. Default path `/tmp/openclaw/openclaw-YYYY-MM-DD.log` (daily rolling, gateway-host TZ); rotates at `logging.maxFileBytes` (default **100 MB**) into `.1`–`.5` archives; falls back to `os.tmpdir()/openclaw-<uid>` (always on Windows).
- **Top-level machine-filterable fields:** `hostname`, `message`, `agent_id`, `session_id`, `channel`.
- **Privacy boundary:** transcript text, audio, turn ids, call ids, and provider item ids are **never** copied into log records.
- **Levels:** `silent|fatal|error|warn|info|debug|trace`; file via `logging.level`, console via `logging.consoleLevel` (default `info`). `--verbose` affects console only. Targeted `agents/harness` debug logging surfaces the structured **`"agent harness selected"`** record (selected id, selection reason, runtime/fallback policy, and — in `auto` mode — each plugin candidate's support result).
- **Redaction:** `logging.redactSensitive` (`off|tools`, default `tools`), `logging.redactPatterns` (regex); masks keeping first 6 + last 4 chars for values ≥18 chars, else `***`. Some surfaces always redact.
- **Console styles:** `pretty|compact|json`; subsystem prefixes like `[gateway]`, `[canvas]`.
- **Tailing:** `openclaw logs --follow`; Control UI Logs tab (`logs.tail` endpoint).

**Session transcripts:** persisted as **JSONL on the gateway host** (`sessions/*.jsonl`). An observability discussion also references `cache-trace.jsonl` + `openclaw.log` for JSONL→SQLite ingestion.

## Where errors can originate (for the trace *reader*)

- **Transcript vs. log split:** the JSONL *log* deliberately omits transcript content and provider item ids — you must read the *session transcript* (`sessions/*.jsonl`) to reconstruct the actual model/tool exchange. Reading only the log will miss the substance of a run.
- **Redaction gaps in analysis:** `redactSensitive=tools` masks values ≥18 chars — a masked argument/result can hide the exact input that triggered a failure.
- **Rotation loss:** at 100 MB the log rotates and only `.1`–`.5` are kept; older evidence for long runs may be gone.
- **The `"agent harness selected"` record** is the single best anchor for attributing a run to a runtime/harness and understanding why fallback did/didn't happen — enable `agents/harness` debug logging to get it.

## Sources

- `https://docs.openclaw.ai/gateway/logging` — JSONL format, path, rotation, fields, levels, redaction, `"agent harness selected"` record.
- `https://github.com/openclaw/openclaw/blob/main/docs/logging.md` — repo logging reference.

> Gap: the full per-line **session-transcript JSONL schema** (`sessions/*.jsonl`) is not spelled out in the two logging doc pages fetched — confirm against the repo before relying on exact field names.

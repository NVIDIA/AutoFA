# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers the corpus readers share; they work on docs handed in, never address files."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from evalrdl_readers.core import Doc, Site, Unreadable
from evalrdl_schema.vocab import CURRENT as VOCAB

# PyYAML is a dependency of these readers, not of the core.
try:  # pragma: no cover - environment dependent
    import yaml as _yaml
except ImportError:  # pragma: no cover
    _yaml = None

#: Why `.intake-manifest.json` is ignored in every reader: it is the intake transport's
#: receipt of the copy it made, a fact about the service and not about the run.
INTAKE_RECEIPT = (
    "the intake transport's receipt of the copy it made; reading it would read the service, "
    "not the run"
)

_TOP_SCALAR = re.compile(r"^([A-Za-z_][A-Za-z0-9_.-]*):[ \t]*(.*)$")


def yaml_doc(doc: Doc | None) -> dict:
    """Parse a YAML document, or return {} and let the caller's declared count complain."""
    if doc is None:
        return {}
    text = doc.text()
    if _yaml is not None:
        try:
            out = _yaml.safe_load(text)
        except Exception:
            out = None
        if isinstance(out, dict):
            return out
    # No partial fallback parse: a partial dict looks like a whole one.
    return {}


def dig(obj: Any, *path: str, default: Any = None) -> Any:
    """Walk a mapping by key path, returning `default` at the first thing that is not one."""
    cur = obj
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def records(doc: Doc, question: str) -> Iterable[Any]:
    """Every object in a JSONL file with its line number, and an `Unreadable` per bad line.

        for item in records(doc, "rollout.score"):
            if isinstance(item, Unreadable):
                yield item
                continue
            line, rec = item
    """
    for line, rec, err in doc.jsonl():
        if err:
            yield Unreadable(
                question=question,
                where=f"{doc.relpath}:{line}",
                detail=f"{err} — the record is lost, not merely unread",
            )
            continue
        yield line, rec


def digest(doc: Doc) -> str:
    """sha256 of a file's bytes — the only way to say `these two paths are one file`."""
    return hashlib.sha256(doc.bytes()).hexdigest()


def site_key(site: Site) -> str:
    """A rollout key independent of nesting depth: the enclosing site names, root dropped.

    Stays unique when a host holds two harbor jobs.
    """
    return "/".join(k for k in site.chain if k)


def as_float(value: Any) -> float | None:
    """A number, or None. A reward written as the string "0.9000" is still a number."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def dumps(value: Any, limit: int | None = None) -> str:
    """`value` as JSON. Whole unless a caller asks for a preview: stored content is never cut."""
    text = json.dumps(value, default=str)
    return text if limit is None else text[:limit]


#: Characters of a tool call's arguments kept in `args_head`, the narrow column a trace renders.
ARGS_HEAD_CHARS = 200


def args_text(args: Any, limit: int | None = None) -> str:
    """A tool call's arguments as text: a string as written, else its JSON; "" for none."""
    return "" if args is None else args if isinstance(args, str) else dumps(args, limit)


def tool_call(name: Any, call_id: Any, args: str, **extra) -> dict:
    """One row of `tool_calls`; `args` is the arguments' text, kept whole and as a head."""
    return {
        "tool_name": name,
        "call_id_source": call_id,
        "args_chars": len(args),
        "args_head": args[:ARGS_HEAD_CHARS],
        "args_full": args,
        **extra,
    }


#: Suffixes nemo-gym derives from its output stem. The stem
#: is set by the caller, so it is discovered from the files present, never matched literally.
FAILURES = "_failures"
MATERIALIZED_INPUTS = "_materialized_inputs"
AGGREGATE_METRICS = "_aggregate_metrics"
MULTISTAGE_STATE = "_multistage_state"
SIDECAR_SUFFIXES = (FAILURES, MATERIALIZED_INPUTS, AGGREGATE_METRICS, MULTISTAGE_STATE)


def is_sidecar_of(name: str, stem: str) -> bool:
    base = name.rsplit(".", 1)[0]
    return any(base == f"{stem}{suf}" for suf in SIDECAR_SUFFIXES)


#: Keys a nemo-gym failure record writes its message under; a record carries exactly one.
FAILURE_MESSAGE_KEYS = ("error_message", "_ng_failure_message", "_ng_failure_judge_error",
                        "failure_reason", "error")


def failure_message(rec: dict) -> str | None:
    """What one nemo-gym failure record says went wrong, under whichever key wrote it."""
    for key in FAILURE_MESSAGE_KEYS:
        value = rec.get(key)
        if value is not None:
            return value
    return None


#: Keys a nemo-gym failure record writes its reward under. A failed rollout can carry a
#: passing verifier reward (the runner errored after the verifier passed).
FAILURE_REWARD_KEYS = ("reward", "verifier_reward")


def failure_reward(rec: dict):
    """What one nemo-gym failure record scored, under whichever key its runner wrote."""
    for key in FAILURE_REWARD_KEYS:
        value = rec.get(key)
        if value is not None:
            return as_float(value)
    return None


#: The nemo-gym reserved key only the rollout stream carries; tells the stream from
#: rollout_verdicts.jsonl, which shares the index keys.
STREAM_MARK = "_ng_rollout_id"


def carries_stream_mark(doc: Doc) -> bool:
    """Does this file's first record look like a nemo-gym rollout row?"""
    for _line, rec, err in doc.jsonl():
        return not err and isinstance(rec, dict) and STREAM_MARK in rec
    return False


#: A harbor trial's `artifacts/manifest.json`: the collector's index of what it copied,
#: not a payload.
ARTIFACT_INDEX = "manifest.json"


def artifact_row(doc: Doc, **extra) -> dict:
    """One artifact, in the store's column names.

    `storage=external` because the store did not copy the bytes; `path` is
    where the producer wrote it, `uri` a `file://` locator.
    """
    return {
        "name": doc.name,
        "path": doc.relpath,
        "size_bytes": doc.size,
        "sha256": digest(doc),
        "origin": "file_tree",
        "storage": "external",
        "uri": f"file://{doc.relpath}",
        **({"text": doc.text(limit=65536)} if _is_text(doc.name) else {}),
        **artifact_type(doc.name),
        **extra,
    }


#: Extension -> (kind, mime) in production's `artifacts.kind` vocabulary; unknown is `other`.
ARTIFACT_TYPE: dict[str, tuple[str, str]] = {
    "log": ("log", "text/plain"), "txt": ("log", "text/plain"),
    "json": ("data", "application/json"), "jsonl": ("data", "application/jsonl"),
    "yaml": ("data", "application/yaml"), "yml": ("data", "application/yaml"),
    "csv": ("data", "text/csv"), "xlsx": ("data",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "png": ("image", "image/png"), "jpg": ("image", "image/jpeg"),
    "jpeg": ("image", "image/jpeg"), "gif": ("image", "image/gif"),
    "svg": ("image", "image/svg+xml"),
    "py": ("code", "text/x-python"), "sh": ("code", "application/x-sh"),
    "patch": ("code", "text/x-diff"), "diff": ("code", "text/x-diff"),
    "pdf": ("doc", "application/pdf"), "md": ("doc", "text/markdown"),
    "docx": ("doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "pptx": ("doc",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
}


#: Suffixes whose bytes are human-readable text and are stored as `text`.
TEXT_SUFFIXES = frozenset({"log", "txt", "json", "jsonl", "yaml", "yml", "csv", "md",
                           "py", "sh", "patch", "diff", "svg"})


def _is_text(name: str) -> bool:
    return name.rsplit(".", 1)[-1].lower() in TEXT_SUFFIXES if "." in name else False


def artifact_type(name: str) -> dict:
    kind, mime = ARTIFACT_TYPE.get(name.rsplit(".", 1)[-1].lower() if "." in name else "",
                                   ("other", None))
    return {"kind": kind, "mime": mime}


#: The store's closed event vocabulary.
EVENT_VOCAB = VOCAB.words["event_kind"]

#: Producer spellings -> event vocabulary. Vocabulary words pass through unmapped; other
#: unmapped spellings become `other` with `kind_raw`, as production's `escape()` does.
EVENT_KIND: dict[str, str] = {
    "assistant": "message",
    "user": "message",
    "thinking": "reasoning",
    "function_call": "tool_call",
    "function_call_output": "tool_result",
    "observation": "tool_result",
    "tool": "tool_result",
}


def event_kind(raw) -> dict:
    """`{kind, kind_raw}` -- production's `escape()`, spelt as the two columns it fills."""
    k = EVENT_KIND.get(raw or "", raw)
    return {"kind": k, "kind_raw": None} if k in EVENT_VOCAB else {"kind": "other", "kind_raw": raw}


def primary_stream(docs) -> Doc | None:
    """Which of these `.jsonl` files is the rollout stream, by the sidecar relationship.

    Lives here because `At` cannot express an address computed from another file or an
    exclusion. Loose files with no sidecar are settled by `STREAM_MARK`.
    """
    names = {d.name.rsplit(".", 1)[0]: d for d in docs}
    with_family = [
        d
        for stem, d in names.items()
        if any(is_sidecar_of(other.name, stem) for other in docs if other is not d)
    ]
    if len(with_family) == 1:
        return with_family[0]
    loose = [
        d
        for stem, d in names.items()
        if not any(is_sidecar_of(d.name, s) for s in names if s != stem)
    ]
    marked = [d for d in loose if carries_stream_mark(d)]
    loose = marked or loose
    return loose[0] if len(loose) == 1 else None


#: The store's closed part vocabularies, `evalrdl.intake.model.Part`.
PART_VOCAB = VOCAB.words["part_type"]

#: Producer content-block spellings -> part vocabulary.
PART_TYPE: dict[str, str] = {
    "output_text": "text",
    "input_text": "text",
    "text": "text",
    "summary_text": "text",
    "input_image": "image",
    "output_image": "image",
    "image_url": "image",
    "input_audio": "binary",
    "input_file": "binary",
    # A base64-embedded file (e.g. a PDF) handed to the model.
    "document": "binary",
    "refusal": "text",
}
CHANNEL_VOCAB = frozenset({"content", "reasoning", "tool_result"})


def part(text, *, channel="content", origin="body", part_type="text", **extra) -> dict:
    """One part of one event, in `parts`/`part_texts` columns; `chars` survives text drops."""
    body = "" if text is None else str(text)
    kind = PART_TYPE.get(part_type, part_type)
    return {
        "part_type": kind if kind in PART_VOCAB else "other",
        "part_type_raw": None if kind in PART_VOCAB else part_type,
        "channel": channel if channel in CHANNEL_VOCAB else "other",
        "origin": origin,
        "chars": len(body),
        "text": body,
        **extra,
    }


def inline_artifact(url: str) -> dict | None:
    """A base64 data URL inlined in a content block, as an artifact row; None if undecodable."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    head, _, payload = url.partition(",")
    if not payload or ";base64" not in head:
        return None
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:
        return None
    mime = head[len("data:"):].split(";", 1)[0] or None
    ext = (mime or "").rpartition("/")[2] or "bin"
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "name": f"{digest[:16]}.{ext}",
        "mime": mime,
        "size_bytes": len(raw),
        "sha256": digest,
        "origin": "inline",
        "storage": "inline",
        "kind": artifact_type(f"x.{ext}")["kind"],
        "text": url,
    }


#: Content blocks whose `text` is the whole of what they say (no other key); any other
#: mapping is kept whole as JSON.
_TEXT_BLOCKS = (None, "text", "input_text", "output_text")


def text_parts(value, *, channel="content", origin="body") -> list[dict]:
    """Parts of a text value: none if absent or empty, one per string, recursive for lists, JSON otherwise."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part(value, channel=channel, origin=origin)]
    if isinstance(value, (list, tuple)):
        return [p for v in value for p in text_parts(v, channel=channel, origin=origin)]
    if (isinstance(value, dict) and isinstance(value.get("text"), str)
            and value.get("type") in _TEXT_BLOCKS and set(value) <= {"type", "text"}):
        return text_parts(value["text"], channel=channel, origin=origin)
    if isinstance(value, dict) and value.get("type") in _IMAGE_BLOCKS:
        return [_image_part(value, channel, origin)]
    return [part(dumps(value), channel=channel, origin=origin)]


#: Content-block types that carry one image, by the spelling each producer uses.
_IMAGE_BLOCKS = frozenset({"image", "input_image", "image_url"})


def _image_part(block: dict, channel: str, origin: str) -> dict:
    """An image content block as an image part, its bytes or its location as the artifact.

    Base64 (Anthropic `source`, or a data URL) is inlined; a `path`/`url` source is an
    external artifact, as the adapter records it.
    """
    src = block.get("source") if isinstance(block.get("source"), dict) else {}
    url = block.get("image_url")
    url = url.get("url") if isinstance(url, dict) else url
    if src.get("type") == "base64" and src.get("data"):
        url = f"data:{src.get('media_type') or 'application/octet-stream'};base64,{src['data']}"
    p = part(None, channel=channel, origin=origin, part_type="image")
    art = inline_artifact(url) if isinstance(url, str) else None
    where = src.get("path") or src.get("url") or (url if isinstance(url, str)
                                                   and not url.startswith("data:") else None)
    if art is None and where:
        art = {"name": None, "kind": "image", "origin": "content_part", "storage": "external",
               "uri": str(where), "mime": src.get("media_type")}
    if art is not None:
        p["artifact"] = art
        if art.get("mime"):
            p["media_type"] = art["mime"]
    return p


_THINK = re.compile(r"</?think>")


def split_inline_think(parts: list) -> list:
    """An assistant turn's content parts with inline `<think>` spans split out as reasoning.

    Tolerates an unterminated `<think>` and an orphan `</think>` (text before it is reasoning).
    """
    out = []
    for p in parts:
        text = p.get("text") if isinstance(p, dict) else None
        if (not isinstance(text, str) or p.get("channel", "content") != "content"
                or not _THINK.search(text)):
            out.append(p)
            continue
        segs, pos, thinking = [], 0, False
        for m in _THINK.finditer(text):
            chunk, closer = text[pos:m.start()], m.group() == "</think>"
            if chunk.strip():
                segs.append((closer, chunk))
            pos, thinking = m.end(), not closer
        if text[pos:].strip():
            segs.append((thinking, text[pos:]))
        if not segs:
            out.append(p)
            continue
        for reasoning, chunk in segs:
            out.append({**p, "text": chunk, "chars": len(chunk), "channel": "reasoning" if reasoning else "content",
                        "origin": "inline_think" if reasoning else p.get("origin", "body")})
    return out


def bookkeeping_event(rec: dict, eid: str, started_at=None, kind=None) -> dict:
    """A record about the harness rather than the conversation, kept whole as JSON."""
    return {
        "source_event_id": eid,
        "started_at": started_at,
        "role": "harness",
        **event_kind(kind or rec.get("type")),
        "parts": [part(dumps(rec), origin="bookkeeping")],
        "tool_calls": [],
    }


def stdout_split(doc: Doc) -> tuple[list[tuple[int, dict]], list[tuple[int, str]]]:
    """An agent's stdout as (line, JSON event) pairs and the (line, text) lines that are not."""
    events, text = [], []
    for line, raw in enumerate(doc.text().split("\n"), 1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            text.append((line, raw))
            continue
        if isinstance(rec, dict):
            events.append((line, rec))
    return events, text


def field_artifact(text: str, *, kind: str, role: str, name: str | None) -> dict:
    """A record field or small document kept as an inline artifact, in the row production
    writes for one: size and digest over the UTF-8 text."""
    data = text.encode()
    return {"name": name, "kind": kind, "role": role, "origin": "source_field",
            "storage": "inline", "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "text": text}


#: Fields that hold what the evaluator expected rather than what the model did, in the token
#: form `reference_token` gives a key; an outcome artifact keeps none of them.
REFERENCE_FIELDS = frozenset({
    "expected_answer", "golden", "golden_patch", "ground_truth", "reference",
    "reference_answer", "reference_patch", "solution", "target",
})


def reference_token(value: object) -> str:
    """A key folded to snake case, so `goldenPatch`, `Golden-Patch` and `golden_patch` are one."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value or "").strip())
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def has_value(value: object) -> bool:
    """Whether a record field holds anything: a blank string is no more a value than null."""
    return value is not None and not (isinstance(value, str) and not value.strip())


def without_reference_fields(value: Any) -> Any:
    """A JSON value with every `REFERENCE_FIELDS` key removed, at any depth."""
    if isinstance(value, dict):
        return {k: without_reference_fields(v) for k, v in value.items()
                if reference_token(k) not in REFERENCE_FIELDS}
    if isinstance(value, list):
        return [without_reference_fields(v) for v in value]
    return value


def record_ref(doc: Doc, line: int) -> str:
    """One record of a stream, named by its file and line. Two readers that read different
    parts of the same record name it alike without knowing each other's fields: the reader
    that owns the record states which rollout it is, the other is joined to it by this name."""
    return f"{doc.path}:{line}"

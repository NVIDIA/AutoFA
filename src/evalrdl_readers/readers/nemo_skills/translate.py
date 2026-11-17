# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from evalrdl_readers.core import Answer, Sources, Unreadable

from ..common import as_float, event_kind, records, text_parts
from ..dialects.chat import chat_messages


def ns_task_dir(src: Sources):
    """The `eval-results/<task>/` directory the run's scores or outputs sit in: the name the
    harness wrote them under, rendered, where run_config may still hold the template."""
    names = sorted({doc.relpath.split("/")[1] for doc in src.all})
    if names:
        yield Answer("run.benchmark", names[0],
                     note=None if len(names) == 1 else f"{len(names)} task directories: {names}")


def _metric_rows(doc, where: str):
    body = doc.json()
    if not isinstance(body, dict):
        return
    for name, block in body.items():
        if not isinstance(block, dict):
            continue
        for agg, values in block.items():
            if not isinstance(values, dict):
                continue
            yield {
                "key": f"{name}/{agg}",
                "subject": name,
                "aggregate": agg,
                "num_entries": values.get("num_entries"),
                "avg_tokens": values.get("avg_tokens"),
                "score": as_float(
                    values.get("symbolic_correct")
                    if values.get("symbolic_correct") is not None
                    else values.get("accuracy")
                ),
            }


def ns_task_metrics(src: Sources):
    """The parent metrics.json; lands beside its subtasks' rows, since it re-reports them."""
    for row in _metric_rows(src.doc(0), "task"):
        yield Answer("run.task_metrics", row, input_key=row["key"])


def ns_subtask_metrics(src: Sources):
    for row in _metric_rows(src.doc(0), "subtask"):
        yield Answer("run.task_metrics", row, input_key=row["key"])


def _repeat_of(name: str) -> int:
    """`output-rs<N>.jsonl` -> N, `output.jsonl` -> 0; the record itself carries no repeat index."""
    stem = name.rsplit(".", 1)[0]
    if stem.startswith("output-rs") and stem[len("output-rs") :].isdigit():
        return int(stem[len("output-rs") :])
    return 0


#: Fields that name the task, probed in order.
_ID_FIELDS = ("id", "problem_id", "problem_idx", "question_id", "task_id",
              "uid", "key", "index", "Unnamed: 0")
_PROMPT_FIELDS = ("problem", "question", "query", "prompt", "problem_description_main",
                  "question_content", "text")
_ANSWER_FIELDS = ("generation", "response")
#: Judge-graded families write the judge's reply under these names.
_JUDGE_FIELDS = ("judgement", "judgement-gen-base", "judgement-base-gen")
_VERDICT = {"A": 1.0, "B": 0.0, "C": 0.5, "D": 0.0, "CORRECT": 1.0, "INCORRECT": 0.0,
            "YES": 1.0, "NO": 0.0, "TRUE": 1.0, "FALSE": 0.0}


def _task_of(rec: dict) -> str | None:
    for f in _ID_FIELDS:
        v = rec.get(f)
        if v is not None:
            sub = rec.get("sub_query_id")
            return f"{v}#{sub}" if sub not in (None, 0) else str(v)
    return None


def _score(rec: dict) -> float | None:
    """The first verdict shape this record carries, per benchmark family; None when unscored."""
    for f in ("symbolic_correct", "is_correct", "res", "correct", "execution_match", "passed"):
        v = rec.get(f)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
    for f in ("comet", "score"):
        v = rec.get(f)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    for f in ("strict_eval", "loose_eval"):  # ifeval: strict is the headline number
        e = rec.get(f)
        if isinstance(e, dict) and isinstance(e.get("follow_all_instructions"), bool):
            return 1.0 if e["follow_all_instructions"] else 0.0
    gl = rec.get("graded_list")  # livecodebench: one bool per test
    if isinstance(gl, list) and gl and all(isinstance(x, bool) for x in gl):
        return sum(gl) / len(gl)
    es = rec.get("eval_status")  # scicode: one status per sub-step
    if isinstance(es, list) and es:
        return sum(isinstance(x, dict) and x.get("process_status") == "completed" for x in es) / len(es)
    j = rec.get("judgement")
    if isinstance(j, str) and j.strip():
        v = j.strip().strip("`").strip()
        verdict = v.splitlines()[-1].strip().upper() if v else ""
        exp = rec.get("expected_judgement")
        if isinstance(exp, str) and exp.strip():
            return 1.0 if verdict == exp.strip().upper() else 0.0
        return _VERDICT.get(verdict)
    return None


def _outputs(src: Sources, where: str):
    site = src.site.path.rsplit("/", 1)[-1]
    for doc in src.docs(0):
        rep = _repeat_of(doc.name)
        seen: set[str] = set()
        for item in records(doc, "rollout.score"):
            if isinstance(item, Unreadable):
                yield item
                continue
            line, rec = item
            task = _task_of(rec)
            # A repeated id within one file would merge two rollouts; the line keeps them apart.
            idx = task if task is not None and task not in seen else f"line:{line}"
            seen.add(idx)
            yield (f"{site}/{rep}/{idx}",), rep, rec, _score(rec)


def ns_per_sample(src: Sources):
    """The per-sample output directory beside eval-results/, one `output-rs<N>.jsonl` per repeat.

    Separate input because it hangs off a different landmark.
    """
    for doc in src.docs(0):
        rep = _repeat_of(doc.name)
        for item in records(doc, "rollout.score"):
            if isinstance(item, Unreadable):
                yield item
                continue
            line, rec = item
            idx = rec.get("problem_idx")
            key = (f"per-sample/{rep}/{idx if idx is not None else line}",)
            yield Answer("rollout.score", _score(rec), key=key)
            yield from _ns_facts(rec, key, rep)
            yield from _ns_events(rec, key)


def _ns_facts(rec: dict, key: tuple, rep):
    """Everything one output.jsonl record says, shared by the task and subtask streams."""
    yield Answer("rollout.repeat", rep, key=key)
    yield Answer("rollout.task", _task_of(rec), key=key)
    yield Answer("rollout.stop_reason", rec.get("finish_reason"), key=key)
    prompt = rec.get("num_input_tokens")
    if prompt is None:
        prompt = rec.get("num_prompt_tokens")
    completion = rec.get("num_generated_tokens")
    yield Answer("rollout.n_prompt_tokens", prompt, key=key)
    yield Answer("rollout.n_completion_tokens", completion, key=key)
    if isinstance(prompt, int) and isinstance(completion, int):
        yield Answer("rollout.n_tokens", prompt + completion, key=key)
    yield Answer("rollout.started_at", rec.get("generation_start_time"), key=key)
    yield Answer("rollout.finished_at", rec.get("generation_end_time"), key=key)


def _ns_events(rec: dict, key: tuple):
    """The transcript of a generation record, and the judge's reply in a lane of its own.

    Prefers `conversation` (OpenAI messages with tool exchanges). Otherwise the asked turn and
    one answered turn: reasoning plus `generation`/`response`. `serialized_output` is not the
    answer where a judge graded the record: there it holds the judge's reply.
    """
    conversation = rec.get("conversation")
    if conversation:
        turns = chat_messages(conversation)
        said = [p["text"] for t in turns for p in t["parts"]]
        gen = rec.get("generation")
        if gen and gen not in said:
            turns.append({"role": "assistant", "kind": "message",
                          "parts": text_parts(gen), "tool_calls": [], "answers": []})
    else:
        prompt = next((rec[f] for f in _PROMPT_FIELDS if rec.get(f)), None)
        answer = next((rec[f] for f in _ANSWER_FIELDS if rec.get(f) is not None), None)
        so = rec.get("serialized_output")
        if answer is None and not any(rec.get(f) for f in _JUDGE_FIELDS) and isinstance(so, list):
            answer = next((m.get("content") for m in reversed(so)
                           if isinstance(m, dict) and m.get("role") == "assistant"), None)
        turns = []
        if prompt is not None:
            turns.append({"role": "user", "kind": "message", "parts": text_parts(prompt),
                          "tool_calls": [], "answers": []})
        said = (text_parts(rec.get("reasoning_content"), channel="reasoning")
                + text_parts(answer))
        if said or answer is not None:
            turns.append({"role": "assistant", "kind": "message", "parts": said,
                          "tool_calls": [], "answers": [], "finish_reason": rec.get("finish_reason")})
    for i, t in enumerate(turns):
        yield Answer(
            "rollout.events",
            {
                "source_event_id": f"turn:{i:09d}",
                "read_seq": f"{i:09d}",
                "started_at": rec.get("generation_start_time") if i == 0 else None,
                "role": t["role"],
                **event_kind(t["kind"]),
                **({"finish_reason": t["finish_reason"]} if t.get("finish_reason") else {}),
                "parts": t["parts"],
                "tool_calls": t["tool_calls"],
                **({"answers_call_ids": t["answers"]} if t["answers"] else {}),
            },
            key=key,
            input_key=f"turn:{i:09d}",
        )
    for i, f in enumerate(f for f in _JUDGE_FIELDS if rec.get(f)):
        yield Answer(
            "rollout.events",
            {
                "source_event_id": f,
                "read_seq": f"{i:09d}",
                "role": "judge",
                **event_kind("message"),
                "parts": text_parts(rec[f]),
            },
            key=key,
            input_key=f"judge:{f}",
            lane=("judge", ""),
        )


def ns_outputs(src: Sources):
    for item in _outputs(src, "task"):
        if isinstance(item, Unreadable):
            yield item
            continue
        key, rep, rec, correct = item
        yield Answer("rollout.score", correct, key=key)
        yield from _ns_facts(rec, key, rep)
        yield from _ns_events(rec, key)


def ns_subtask_outputs(src: Sources):
    for item in _outputs(src, "subtask"):
        if isinstance(item, Unreadable):
            yield item
            continue
        key, rep, rec, correct = item
        yield Answer("rollout.score", correct, key=key)
        yield from _ns_facts(rec, key, rep)
        yield from _ns_events(rec, key)

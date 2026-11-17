# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A rollout is one task and one repeat; `agent_ref` labels it. `convert` refuses a read in which
two agents land on one task and repeat, instead of storing two rollouts under one address."""

from __future__ import annotations

import json
import sys

import pytest

from evalrdl_readers.convert import DuplicateRollout, convert
from evalrdl_readers.core import Answer, At, Card, Grain, Input, Landmark, Population, Reader, Rows
from evalrdl_readers.readers import questions as Q
from evalrdl_readers.readers.store_schema import READER_SCHEMA

RUN = Landmark(name="run", marker="samples.json", where=("",),
               expect=Card.ONE(because="one samples file per run"))  # fmt: skip


def samples(src):
    for s in src.doc(0).json():
        key = (s["key"],)
        if s.get("task"):
            yield Answer("rollout.task", s["task"], key=key)
        yield Answer("rollout.repeat", s["repeat"], key=key)
        yield Answer("rollout.agent_name", s["agent"], key=key)
        yield Answer("rollout.score", s["score"], key=key)


SAMPLES = Input(
    name="samples", via=samples, reads=[At(RUN, "samples.json")],
    answers={
        "rollout.task": Card.ANY(because="a sample may not name its task"),
        "rollout.repeat": Card.AT_LEAST_ONE(because="every sample has a repeat"),
        "rollout.agent_name": Card.AT_LEAST_ONE(because="every sample names its agent"),
        "rollout.score": Card.AT_LEAST_ONE(because="every sample is scored"),
    },
)  # fmt: skip

READER = Reader(
    schema=READER_SCHEMA,
    counts=Population.COMPLETED,
    name="samples",
    landmarks=(RUN,),
    inputs=(("samples", (SAMPLES,)),),
    rows=(Rows(question=Grain.ROLLOUT, may_create=("samples",),
               because="a sample exists because the file lists it"),),
    questions=(Q.ROLLOUT_TASK, Q.ROLLOUT_REPEAT, Q.ROLLOUT_AGENT_NAME, Q.ROLLOUT_SCORE),
)  # fmt: skip


def _convert(tmp_path, rows):
    (tmp_path / "samples.json").write_text(json.dumps(rows))
    result = READER.read(str(tmp_path))
    return convert(result, path=str(tmp_path), label="t", readers=(READER,))


def test_two_agents_on_one_task_and_repeat_are_refused(tmp_path):
    # `t1` names no task, so its key is its address, and that is the task `a` names.
    rows = [
        dict(key="a", task="t1", repeat=0, agent="solver", score=1.0),
        dict(key="t1", repeat=0, agent="critic", score=0.0),
    ]
    with pytest.raises(DuplicateRollout, match="'t1' repeat 0"):
        _convert(tmp_path, rows)


def test_each_rollout_keeps_its_agent_as_a_label(tmp_path):
    rows = [
        dict(key="a", task="t1", repeat=0, agent="solver", score=1.0),
        dict(key="b", task="t1", repeat=1, agent="critic", score=0.0),
    ]
    got = _convert(tmp_path, rows).tables["rollouts"]
    assert sorted((r["task_id"], r["repeat"], r["agent_ref"]) for r in got) == [
        ("t1", 0, "solver"), ("t1", 1, "critic")]  # fmt: skip


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))

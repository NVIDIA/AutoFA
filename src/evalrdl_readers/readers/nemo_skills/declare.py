# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from evalrdl_readers.core import (
    ROOT,
    At,
    Card,
    Contribute,
    Fold,
    Grain,
    Ignore,
    Input,
    Lane,
    Merge,
    Op,
    Order,
    Population,
    Question,
    Reader,
    Rows,
)
from evalrdl_readers.readers.nemo_skills.landmarks import RUN, SUBTASK, TASK
from evalrdl_readers.readers.nemo_skills.translate import (
    ns_outputs,
    ns_per_sample,
    ns_subtask_metrics,
    ns_subtask_outputs,
    ns_task_dir,
    ns_task_metrics,
)

from .. import questions as Q
from ..common import INTAKE_RECEIPT
from ..store_schema import READER_SCHEMA

RUN_TASK_METRICS = Question(
    name="run.task_metrics", grain=Grain.RUN, merge=Merge.UNION, lands="runs.task_metrics",
    means=(
        "every published metric at both levels, the parent aggregate and the subtask "
        "aggregates it comes from; which one is the run's score is a decision, so both are "
        "stored and neither chosen"
    ),
)

NS_TASK_DIR = Input(
    name="ns_task_dir", via=ns_task_dir,
    reads=[
        At(RUN, "eval-results/*/metrics.json"), At(RUN, "eval-results/*/*/metrics.json"),
        At(RUN, "eval-results/*/output.jsonl"), At(RUN, "eval-results/*/output-rs*.jsonl"),
        At(RUN, "eval-results/*/*/output.jsonl"), At(RUN, "eval-results/*/*/output-rs*.jsonl"),
    ],
    answers={
        "run.benchmark": Card.ANY(
            because=(
                "the directory the harness wrote the task's results under: the task name "
                "rendered, where the wrapper's config can hold the unrendered template"
            ),
        ),
    },
)

NS_TASK_METRICS = Input(
    name="ns_task_metrics", via=ns_task_metrics, reads=[At(TASK, "metrics.json")],
    answers={"run.task_metrics": Card.AT_LEAST_ONE(because="the task directory exists because nemo-skills wrote metrics into it")},
)

NS_SUBTASK_METRICS = Input(
    name="ns_subtask_metrics", via=ns_subtask_metrics, reads=[At(SUBTASK, "metrics.json")],
    answers={"run.task_metrics": Card.AT_LEAST_ONE(because="a subtask directory is marked by its own `metrics.json`")},
)

NS_PER_SAMPLE = Input(
    name="ns_per_sample", via=ns_per_sample,
    reads=[At(RUN, "nemo-evaluator-per-sample-outputs/output-rs*.jsonl")],
    answers={
        "rollout.score": Card.ANY(because="only some runs write the per-sample output directory"),
        "rollout.task": Card.ANY(because="same file, same absence"),
        "rollout.repeat": Card.ANY(because="same directory, same absence; the repeat is in the filename"),
        "rollout.stop_reason": Card.ANY(because="same directory, same absence"),
        "rollout.n_tokens": Card.ANY(because="same directory, same absence; the sum of the two halves where both are written"),
        "rollout.n_prompt_tokens": Card.ANY(because="same directory, same absence; these records carry `num_prompt_tokens`"),
        "rollout.n_completion_tokens": Card.ANY(because="same directory, same absence; `num_generated_tokens`"),
        "rollout.started_at": Card.ANY(because="same directory, same absence"),
        "rollout.finished_at": Card.ANY(because="same directory, same absence"),
        "rollout.events": Card.ANY(because="same directory, same absence"),
    },
)

NS_OUTPUTS = Input(
    name="ns_outputs", via=ns_outputs, reads=[At(TASK, "output*.jsonl")],
    answers={
        "rollout.score": Card.ONE(because="a line in output.jsonl is one scored generation"),
        "rollout.task": Card.ANY(because="the task's own id is whichever of `id`, `problem_id`, `problem_idx`, `question_id`, `task_id`, `uid`, `key`, `index` the family writes, and some families write none of them"),
        "rollout.repeat": Card.ONE(because="the repeat comes from the filename, so every row has one"),
        "rollout.stop_reason": Card.ANY(because="`finish_reason` is written only by some families"),
        "rollout.n_tokens": Card.ANY(because="the sum of prompt and completion, only where the record states both"),
        "rollout.n_prompt_tokens": Card.ANY(because="a record writes `num_input_tokens` or `num_prompt_tokens`, never both, and many families write neither"),
        "rollout.n_completion_tokens": Card.ONE(because="`num_generated_tokens`, the tokens the model generated, is on every record; it is the completion, not the total"),
        "rollout.started_at": Card.ANY(because="`generation_start_time` is written only by families whose sampler stamps a clock"),
        "rollout.finished_at": Card.ANY(because="`generation_end_time`, written by the same families"),
        "rollout.events": Card.AT_LEAST_ONE(because="every record carries the asked prompt, so a record has at least one turn"),
    },
)

NS_SUBTASK_OUTPUTS = Input(
    name="ns_subtask_outputs", via=ns_subtask_outputs, reads=[At(SUBTASK, "output*.jsonl")],
    answers={
        "rollout.score": Card.ANY(
            because=(
                "a subtask's `output.jsonl` can be absent while its directory and "
                "`metrics.json` are present"
            ),
        ),
        "rollout.task": Card.ANY(because="same file, same absence"),
        "rollout.repeat": Card.ANY(because="same file, same absence"),
        "rollout.stop_reason": Card.ANY(because="same file, same absence"),
        "rollout.n_tokens": Card.ANY(because="same file, same absence"),
        "rollout.n_prompt_tokens": Card.ANY(because="same file, same absence"),
        "rollout.n_completion_tokens": Card.ANY(because="same file, same absence"),
        "rollout.started_at": Card.ANY(because="same file, same absence"),
        "rollout.finished_at": Card.ANY(because="same file, same absence"),
        "rollout.events": Card.ANY(because="same file, same absence"),
    },
)

READER = Reader(
    schema=READER_SCHEMA,
    counts=Population.COMPLETED,  # an output*.jsonl line is written when a sample finishes
    name="nemo-skills",
    landmarks=(RUN, TASK, SUBTASK),
    inputs=(
        ("nemo-skills",    (NS_TASK_DIR, NS_TASK_METRICS, NS_SUBTASK_METRICS,
                            NS_OUTPUTS, NS_SUBTASK_OUTPUTS, NS_PER_SAMPLE)),
    ),
    questions=(
        Q.ROLLOUT_N_PROMPT_TOKENS, Q.ROLLOUT_N_COMPLETION_TOKENS, Q.ROLLOUT_STARTED, Q.ROLLOUT_FINISHED, Q.ROLLOUT_EVENTS,
        Q.RUN_BENCHMARK, Q.RUN_N_SCORED,
        RUN_TASK_METRICS,
        Q.ROLLOUT_SCORE, Q.ROLLOUT_REPEAT, Q.ROLLOUT_STOP, Q.ROLLOUT_N_TOKENS, Q.ROLLOUT_TASK,
    ),
    lanes=(
        Lane(kind="judge", order_by="read_seq", identity="source_event_id", actor="judge",
             because=(
                 "a judge-graded family asks a second model whether the first one's answer is "
                 "right, and its reply is a turn about the finished conversation, not a turn in "
                 "it. In these families `serialized_output` holds the judge's reply too, so the "
                 "answer is read from `generation`"
             )),
    ),
    rows=(
        Rows(
            question=Grain.ROLLOUT,
            may_create=("ns_outputs", "ns_subtask_outputs", "ns_per_sample"),
            because=(
                "a benchmark writes output files either at the task level or one level "
                "down in subtask directories, never both; either is the enumeration, and "
                "declaring it at the grain backs the key so a new question needs no new line"
            ),
        ),
    ),
    orders=(
        Order(
            question="rollout.score", order=("ns_outputs", "ns_subtask_outputs", "ns_per_sample"),
            keep_loser=True,
            because=(
                "unreachable: a rollout key carries the directory it came from, so a "
                "parent-level row and a subtask-level row never share one; the parent is "
                "ranked first because that is where a flat benchmark writes"
            ),
        ),
        Order(question="rollout.repeat", order=("ns_outputs", "ns_subtask_outputs", "ns_per_sample"),
              keep_loser=True, because="same sources, same disjoint keys, same reason"),
        Order(question="rollout.stop_reason", order=("ns_outputs", "ns_subtask_outputs", "ns_per_sample"),
              keep_loser=True, because="same sources, same disjoint keys, same reason"),
        Order(question="rollout.n_tokens", order=("ns_outputs", "ns_subtask_outputs", "ns_per_sample"),
              keep_loser=True, because="same sources, same disjoint keys, same reason"),
        Order(question="rollout.n_prompt_tokens", order=("ns_outputs", "ns_subtask_outputs", "ns_per_sample"),
              keep_loser=True, because="same sources, same disjoint keys, same reason"),
        Order(question="rollout.n_completion_tokens", order=("ns_outputs", "ns_subtask_outputs", "ns_per_sample"),
              keep_loser=True, because="same sources, same disjoint keys, same reason"),
        Order(question="rollout.started_at", order=("ns_outputs", "ns_subtask_outputs", "ns_per_sample"),
              keep_loser=True, because="same sources, same disjoint keys, same reason"),
        Order(question="rollout.finished_at", order=("ns_outputs", "ns_subtask_outputs", "ns_per_sample"),
              keep_loser=True, because="same sources, same disjoint keys, same reason"),
        Order(question="rollout.task", order=("ns_outputs", "ns_subtask_outputs", "ns_per_sample"),
              keep_loser=True, because="same sources, same disjoint keys, same reason"),
    ),
    folds=(
        Fold(
            into="run.n_scored", over="rollout.score", op=Op.COUNT,
            expect=Card.ANY(
                because="every rollout row here comes from a scored line, so there is no shortfall to withhold a total for",
            ),
            because=(
                "the enumerated count; a subtask whose `output.jsonl` is absent "
                "contributes nothing to it"
            ),
        ),
    ),
    ignored=(
        Ignore(ROOT, ".intake-manifest.json", because=INTAKE_RECEIPT),
        Ignore(TASK, "output*.jsonl.done",
               because="a zero-byte marker nemo-skills writes beside each output file when that repeat has finished; `ns_outputs` reads the file it marks"),
    ),
    attach=(
        Contribute(to="rollout.events", input="ns_per_sample", order_by="read_seq",
                   identity="source_event_id",
                   because="a generation record has one clock for the whole record and none on its turns, so position within the record is the only order; a record is one asked turn and one answered turn carrying its reasoning as a part"),
        Contribute(to="rollout.events", input="ns_outputs", order_by="read_seq",
                   identity="source_event_id",
                   because="a generation record has one clock for the whole record and none on its turns, so position within the record is the only order; a record is one asked turn and one answered turn carrying its reasoning as a part"),
        Contribute(to="rollout.events", input="ns_subtask_outputs", order_by="read_seq",
                   identity="source_event_id",
                   because="a generation record has one clock for the whole record and none on its turns, so position within the record is the only order; a record is one asked turn and one answered turn carrying its reasoning as a part"),
        Contribute(to="run.task_metrics", input="ns_task_metrics",
                   order_by="key", identity="key",
                   because=(
                       "a published metric is identified by its subject and its aggregate, "
                       "and there is no clock to order it by"
                   )),
        Contribute(to="run.task_metrics", input="ns_subtask_metrics",
                   order_by="key", identity="key",
                   because=(
                       "the same identity on purpose: a parent `metrics.json` holds one key "
                       "per child, so the child's row dedupes against the parent's copy when "
                       "they agree and a disagreement becomes a contested finding"
                   )),
    ),
)

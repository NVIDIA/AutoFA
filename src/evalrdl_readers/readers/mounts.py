# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where one producer runs inside another.

Each reader here is a host with the readers it holds mounted in it. The host and the
mounted readers never import each other; what needs both lives on the `Mount`.

A file several producers wrote is read by each of their readers for its own part. A part that
means something only with another producer's key, such as an agent's session id inside the
launcher's trial record, is a seam input on the `Mount`, filed under the producer that wrote
those bytes.
"""

from __future__ import annotations

import re

from evalrdl_readers.core import (
    ROOT,
    Answer,
    At,
    Card,
    Grain,
    Ignore,
    Input,
    Join,
    Key,
    Merge,
    Mount,
    Order,
    Question,
    Relate,
    Rows,
    Sources,
    Unreadable,
)

from . import (
    bundle_converter,
    codex,
    gym,
    harbor,
    nemo_evaluator,
    nemo_evaluator_launcher,
    nemo_skills,
    nooa_sessions,
    pi,
)
from . import questions as Q
from .common import as_float, dig, records, site_key, yaml_doc
from .dialects.atif import ATIF_CANDIDATES, TRAJECTORY_BECAUSE, atif_documents
from .dialects.pi import stdout_session
from .nooa_sessions.declare import STORE as NOOA_STORE
from .nooa_sessions.translate import root_sessions

# ------------------------------------------------------------ nemo-evaluator around a harness

#: gym's headline metric in the wrapper's flatten of its aggregate.
GYM_HEADLINE_METRIC = "key_metrics/mean/reward"


def gym_benchmark_arg(src: Sources):
    """The benchmark gym was given, inside the wrapper's opaque `extra` block."""
    if not src.has(0):
        return
    benchmark = dig(yaml_doc(src.doc(0)), "config", "params", "extra", "nemo_gym", "benchmark")
    if benchmark is not None:
        yield Answer("run.benchmark", benchmark)


def gym_results_metrics(src: Sources):
    """The wrapper's flatten of gym's aggregate; only the headline metric is taken as the score."""
    if not src.has(0):
        return
    groups = dig(yaml_doc(src.doc(0)), "results", "groups", default={}) or {}
    if not isinstance(groups, dict):
        return
    found = []
    for name, group in groups.items():
        metrics = (group or {}).get("metrics") if isinstance(group, dict) else None
        if not isinstance(metrics, dict):
            continue
        for key, entry in metrics.items():
            if not key.endswith("/" + GYM_HEADLINE_METRIC):
                continue
            scores = (entry or {}).get("scores") if isinstance(entry, dict) else None
            for _sk, score in (scores or {}).items():
                value = as_float((score or {}).get("value"))
                if value is not None:
                    found.append((f"{name}:{key}", value))
    if not found:
        return
    yield Answer(
        "run.score",
        found[0][1] if len(found) == 1 else None,
        note=None
        if len(found) < 2
        else f"{len(found)} agent groups publish {GYM_HEADLINE_METRIC}: "
        f"{', '.join(k for k, _ in found)}; no declaration ranks them",
    )


_NUM_REPEATS = re.compile(r"num_repeats[=:]\s*(\d+)")
_BENCHMARK_DIR = re.compile(r"benchmarks/([\w.-]+)/")


def gym_repeats(src: Sources):
    """gym's repeat count, as a key or a `++num_repeats=` override in the wrapper's command."""
    if not src.has(0):
        return
    m = _NUM_REPEATS.search(src.doc(0).text())
    if m:
        yield Answer("run.n_repeats", int(m.group(1)))


def gym_config_scrape(src: Sources):
    """The last `benchmarks/<x>/` path anywhere in the wrapper's config: a guess, and marked one."""
    if not src.has(0):
        return
    found = _BENCHMARK_DIR.findall(src.doc(0).text())
    if found:
        yield Answer(
            "run.benchmark",
            found[-1],
            note=None if len(set(found)) == 1 else f"paths name {sorted(set(found))}",
        )


GYM_REPEATS = Input(
    name="gym_repeats",
    via=gym_repeats,
    reads=[At(nemo_evaluator.ENVELOPE, nemo_evaluator.RUN_CONFIG)],
    answers={
        "run.n_repeats": Card.ANY(because="absent where the run takes gym's default repeat count")
    },
)

GYM_CONFIG_SCRAPE = Input(
    name="gym_config_scrape",
    via=gym_config_scrape,
    reads=[At(nemo_evaluator.ENVELOPE, nemo_evaluator.RUN_CONFIG)],
    answers={
        "run.benchmark": Card.ANY(
            because=(
                "the config's `benchmarks/<x>/` paths name the library directory the run loaded "
                "its data from; a guess, since distinct evals share a directory, so it "
                "ranks below every declared name and may not take a rename"
            )
        )
    },
)

GYM_BENCHMARK_ARG = Input(
    name="gym_benchmark_arg",
    via=gym_benchmark_arg,
    reads=[At(nemo_evaluator.ENVELOPE, nemo_evaluator.RUN_CONFIG)],
    answers={
        "run.benchmark": Card.ANY(
            because="`config.params.extra.nemo_gym.benchmark` is set only where the wrapper passes gym a benchmark"
        )
    },
)

GYM_RESULTS_METRICS = Input(
    name="gym_results_metrics",
    via=gym_results_metrics,
    reads=[At(nemo_evaluator.ENVELOPE, nemo_evaluator.RESULTS)],
    answers={
        "run.score": Card.ANY(
            because=(
                f"only the launcher shape carries a results block, and only some benchmarks name "
                f"their headline {GYM_HEADLINE_METRIC!r}; nothing in the folder names another"
            )
        )
    },
)

WRAPPER = (
    "nemo-evaluator runs the harness and writes its own files beside the harness's; it is "
    "the outer producer"
)

# ---------------------------------------------------------------------------- composed readers

#: gym's mounts: nemo-evaluator around it, and harbor inside it. A host that mounts more in gym
#: passes these with its own in one `mount` call.
GYM_MOUNTS = (
    Mount(
        nemo_evaluator.READER,
        within=gym.RUN,
        where=("",),
        expect=Card.ANY(because="the envelope is optional: gym also runs without the wrapper"),
        because=WRAPPER,
        wraps=True,
        inputs=(
            ("nemo-gym", (GYM_BENCHMARK_ARG, GYM_CONFIG_SCRAPE, GYM_REPEATS, GYM_RESULTS_METRICS)),
        ),
        questions=(Q.RUN_N_REPEATS,),
        ignored=nemo_evaluator.launcher_ignores(ROOT),
        orders=(
            Order(
                question="run.benchmark",
                order=(
                    "gym_benchmark_arg",
                    "ne_run_config",
                    "ne_export_config",
                    "ne_results_yml",
                    "gym_config_scrape",
                    "gym_dataset_name",
                ),
                keep_loser=True,
                because=(
                    "declared names before guessed ones: the argument gym was given first; the "
                    "launcher's mlflow tag is a name a human chose and can be wrong about what "
                    "ran, so a disagreement is CONTESTED; the config-path scrape and the dataset "
                    "name are inferences, ranked last"
                ),
            ),
            Order(
                question="run.score",
                order=("gym_aggregate", "gym_results_metrics", "gym_shard_metrics"),
                keep_loser=True,
                because=(
                    "gym's aggregate did the arithmetic and the wrapper's flatten copies it; the "
                    "shard runner's metrics are over its own shards"
                ),
            ),
            Order(
                question="run.n_expected",
                order=("gym_verification", "gym_shard_run", "ne_results_yml"),
                keep_loser=True,
                because="gym's own counts of what it attempted; the wrapper's `num_rollouts` is null wherever measured",
            ),
        ),
    ),
    Mount(
        harbor.READER,
        within=gym.RUN,
        where=("harbor/*",),
        expect=Card.ANY(
            because="gym runs harbor only for benchmarks whose agent is a harbor agent, one job per rollout"
        ),
        because="gym's harbor agent runs each rollout as a harbor job named after the rollout's position",
        rollout=Join("rollout_position"),
        drop_run=(
            "a job's run-level figures describe one gym rollout, and a gym run holds many jobs; "
            "the trial beside them carries the rollout's own facts"
        ),
        orders=(
            Order(
                question="rollout.error",
                order=(
                    "gym_failures",
                    "harbor_trial",
                    "harbor_exception",
                    "gym_rollouts",
                    "gym_shards",
                    "gym_combined",
                ),
                keep_loser=True,
                because=(
                    "gym's failures sidecar is its typed record; harbor's typed result and "
                    "traceback describe the rollout's trial; the stream's one-word `error_kind` "
                    "is last"
                ),
            ),
            Order(
                question="rollout.stop_reason",
                order=(
                    "gdpval_finish",
                    "harbor_decision",
                    "gym_rollouts",
                    "gym_combined",
                    "gym_shards",
                ),
                keep_loser=True,
                because="an agent's own finish reason first; the stream's `agent_error_kind` is a class, not the agent's words",
            ),
        ),
    ),
)

GYM = gym.READER.mount(*GYM_MOUNTS)

NEMO_SKILLS = nemo_skills.READER.mount(
    Mount(
        nemo_evaluator.READER,
        within=nemo_skills.RUN,
        where=("",),
        expect=Card.ONE(because="nemo-skills runs only inside the wrapper"),
        because=WRAPPER,
        wraps=True,
        ignored=nemo_evaluator.launcher_ignores(ROOT),
        orders=(
            Order(
                question="run.benchmark",
                order=("ns_task_dir", "ne_run_config", "ne_export_config", "ne_results_yml"),
                keep_loser=True,
                because=(
                    "the task directory is what the harness rendered and wrote under, where the "
                    "wrapper's config can hold the unrendered template; then the task the wrapper "
                    "was given, then the name a human tagged the export with"
                ),
            ),
        ),
    ),
)

# ------------------------------------------- the evaluator's run, and who wrote what inside it

NEL = nemo_evaluator_launcher
NEL_ROW_KEY = nemo_evaluator_launcher.translate._row_key

#: The evaluator's streams enumerate the trials; every reader mounted on it only fills trials
#: in, so each Mount restates it over the merged mounted reader's own.
TRIALS = Rows(
    question=Grain.ROLLOUT,
    may_create=("nel_results", "nel_traj_meta"),
    because=(
        "the evaluator's streams enumerate the trials; a document, a proxy record or an agent "
        "log names one and may fill it in, never add one, so one no trial names is a finding"
    ),
)

#: Token totals, from every producer that states one for a rollout, best first.
TOKEN_SOURCES = (
    tuple(f"harbor_atif_trajectories/{d.name}" for d in ATIF_CANDIDATES)[::-1]
    + tuple(f"harbor_atif_results/{d.name}" for d in ATIF_CANDIDATES)[::-1]
    + tuple(f"nel_own_events/{d.name}" for d in ATIF_CANDIDATES)[::-1]
    + ("nel_traj_meta", "nel_own_events/single-turn")
)
TOKENS_BECAUSE = (
    "; an ATIF document's `final_metrics` is the rollup over the steps stored, whoever wrote the "
    "document, and ranks above the evaluator's `model.tokens` block in the record around it; "
    "a disagreement is contested rather than overwritten"
)


def _first_document(rec: dict) -> dict:
    docs = atif_documents(rec)
    return docs[0] if docs else {}


ROLLOUT_PI_SESSION = Question(
    name="rollout.pi_session",
    grain=Grain.ROLLOUT,
    merge=Merge.ONE,
    lands="rollouts.pi_session",
    means="the pi session a trial ran, as pi's stdout inside the trial's trajectory names it",
)


def pi_in_trajectory(src: Sources):
    """A record three producers wrote: the evaluator's record names the trial and holds the ATIF
    document it wrote around pi, whose first step is pi's stdout, which opens with pi's session.
    The session is pi's; the key is the evaluator's; only the seam has both."""
    if not src.has(0):
        return
    for item in records(src.doc(0), "rollout.pi_session"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        steps = _first_document(rec).get("steps")
        first = steps[0] if isinstance(steps, list) and steps and isinstance(steps[0], dict) else {}
        sid = stdout_session(first.get("message"))
        if sid:
            yield Answer(
                "rollout.pi_session", sid, key=NEL_ROW_KEY(rec, line), keys={"pi_session": sid}
            )


PI_IN_TRAJECTORY = Input(
    name="pi_in_trajectory",
    via=pi_in_trajectory,
    reads=[At(NEL.RUN, "trajectories.jsonl")],
    answers={
        "rollout.pi_session": Card.ANY(
            because="only a pi trial's trajectory opens with pi's stdout; others name no pi session"
        )
    },
    mints=(
        Key(
            "pi_session",
            because="pi's session id, read from pi's stdout in the trial's trajectory",
            optional=True,
        ),
    ),
)

ROLLOUT_CODEX_SESSION = Question(
    name="rollout.codex_session",
    grain=Grain.ROLLOUT,
    merge=Merge.ONE,
    lands="rollouts.codex_session",
    means="the codex session a trial ran, as codex's ATIF document names it",
)


def codex_in_trajectory(src: Sources):
    """A record two producers wrote: the evaluator's record names the trial, and the ATIF
    document in it, harbor's codex agent's, names codex's session as its `session_id`."""
    if not src.has(0):
        return
    for item in records(src.doc(0), "rollout.codex_session"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        doc = _first_document(rec)
        sid = doc.get("session_id") if dig(doc, "agent", "name") == "codex" else None
        if sid:
            sid = str(sid)
            yield Answer(
                "rollout.codex_session",
                sid,
                key=NEL_ROW_KEY(rec, line),
                keys={"codex_session": sid},
            )


CODEX_IN_TRAJECTORY = Input(
    name="codex_in_trajectory",
    via=codex_in_trajectory,
    reads=[At(NEL.RUN, "trajectories.jsonl")],
    answers={
        "rollout.codex_session": Card.ANY(
            because="only a codex trial's ATIF document is codex's; others name no codex session"
        )
    },
    mints=(
        Key(
            "codex_session",
            because="codex's session id, as codex's ATIF document in the trial's trajectory names it",
            optional=True,
        ),
    ),
)

#: Where the evaluator keeps each trial's agent logs, relative to the benchmark directory.
AGENT_LOGS = ("agent_logs/*",)
NOT_THE_DIRECTORY = (
    "Not the directory it sits in: the evaluator names it from task metadata, and on a "
    "267-trial pi run 52 of its 104 surviving stdout logs sit in a directory named for another "
    "repeat. A log no trial names is a finding; a trial whose logs were not kept claims nothing"
)
AGENT_LOGS_KEPT = Card.ANY(
    because="the evaluator keeps an agent's logs for pi and codex trials, one directory each"
)

NEMO_EVALUATOR_LAUNCHER = nemo_evaluator_launcher.READER.mount(
    Mount(
        bundle_converter.READER,
        within=ROOT,
        where=("",),
        expect=Card.ANY(because="only a converted bundle has a converter's files around the run"),
        because=(
            "a converter rewrote another run into the evaluator's layout and left its own files "
            "at the top of the bundle"
        ),
        wraps=True,
        rows=(TRIALS,),
    ),
    Mount(
        harbor.RECORDS,
        within=NEL.RUN,
        where=("",),
        expect=Card.ANY(
            because="the evaluator's records hold harbor's documents only where harbor ran"
        ),
        because=(
            "the evaluator runs harbor per trial and puts the trajectory harbor's agent wrote "
            "into the trial's record"
        ),
        rows=(TRIALS,),
        orders=(
            Order(
                question="rollout.n_prompt_tokens",
                order=TOKEN_SOURCES,
                keep_loser=True,
                because=TRAJECTORY_BECAUSE + TOKENS_BECAUSE,
            ),
            Order(
                question="rollout.n_completion_tokens",
                order=TOKEN_SOURCES,
                keep_loser=True,
                because=TRAJECTORY_BECAUSE + TOKENS_BECAUSE,
            ),
        ),
        attach=(
            Relate(
                frm="rollout.atif_trajectory_record",
                to="rollout.trajectory_record",
                on=Join(space="trajectory_record"),
                cardinality=Card.ONE,
                to_card=None,
                same_rollout=True,
                because=(
                    "a document belongs to the trial whose record holds it; the evaluator says "
                    "which trial each record is, harbor what the document in it says"
                ),
            ),
            Relate(
                frm="rollout.atif_results_record",
                to="rollout.results_record",
                on=Join(space="results_record"),
                cardinality=Card.ONE,
                to_card=None,
                same_rollout=True,
                because="the same, for a document read from `results.jsonl`",
            ),
        ),
    ),
    Mount(
        pi.READER,
        within=NEL.RUN,
        where=AGENT_LOGS,
        expect=AGENT_LOGS_KEPT,
        because="harbor runs pi in each trial's sandbox; the evaluator copies pi's logs out of it",
        rows=(TRIALS,),
        inputs=(("pi", (PI_IN_TRAJECTORY,)),),
        questions=(ROLLOUT_PI_SESSION,),
        attach=(
            Relate(
                frm="rollout.pi_log",
                to="rollout.pi_session",
                on=Join(space="pi_session"),
                cardinality=Card.ONE,
                to_card=None,
                same_rollout=True,
                because="pi's logs belong to the trial whose trajectory names their session. "
                + NOT_THE_DIRECTORY,
            ),
        ),
    ),
    Mount(
        codex.READER,
        within=NEL.RUN,
        where=AGENT_LOGS,
        expect=AGENT_LOGS_KEPT,
        because="harbor runs codex in each trial's sandbox; the evaluator copies codex's logs out of it",
        rows=(TRIALS,),
        inputs=(("codex", (CODEX_IN_TRAJECTORY,)),),
        questions=(ROLLOUT_CODEX_SESSION,),
        attach=(
            Relate(
                frm="rollout.codex_log",
                to="rollout.codex_session",
                on=Join(space="codex_session"),
                cardinality=Card.ONE,
                to_card=None,
                same_rollout=True,
                because="codex's logs belong to the trial whose ATIF document names their session. "
                + NOT_THE_DIRECTORY,
            ),
        ),
    ),
)

# ------------------------------------------------------------- NOOA's session stores in harbor

ROLLOUT_NOOA_SESSION = Question(
    name="rollout.nooa_session",
    grain=Grain.ROLLOUT,
    merge=Merge.ONE,
    lands="rollouts.nooa_session",
    means="the NOOA root session a trial ran, as the store in the trial's own directory names it",
)


def nooa_in_trial(src: Sources):
    """The root session each store in a trial's `agent/sessions` starts, keyed by that trial:
    harbor bind-mounts only the trial's own `agent/` into its container, so a store there is
    the trial's. The trial is harbor's; the root session is NOOA's; only the seam has both."""
    key = (site_key(src.site.anchor),)
    for sid in root_sessions(src):
        yield Answer("rollout.nooa_session", sid, key=key, keys={"nooa_session": sid})


NOOA_IN_TRIAL = Input(
    name="nooa_in_trial",
    via=nooa_in_trial,
    reads=[At(nooa_sessions.SESSIONS, NOOA_STORE)],
    answers={
        "rollout.nooa_session": Card.ANY(
            because="a trial that ran NOOA once starts one root session; one whose store is unreadable names none"
        )
    },
    mints=(
        Key(
            "nooa_session",
            because="NOOA's root session id, read from the store in the trial's agent directory",
            optional=True,
        ),
    ),
)

#: harbor read as a job of its own, with NOOA's session stores mounted in each trial that kept them.
HARBOR = harbor.STANDALONE.mount(
    Mount(
        nooa_sessions.MOUNTED,
        within=harbor.TRIAL,
        where=("agent/sessions",),
        expect=Card.ANY(
            because=(
                "NOOA's harbor adapter points `NOOA_SESSIONS_DIR` at `agent/sessions`, and only "
                "its atom agent type writes a store there"
            )
        ),
        because="NOOA's harbor runner keeps its session stores in the trial's own agent directory",
        inputs=(("nooa", (NOOA_IN_TRIAL,)),),
        questions=(ROLLOUT_NOOA_SESSION,),
        ignored=(
            Ignore(
                harbor.TRIAL,
                "agent/traces/*.jsonl",
                because=(
                    "NOOA's runner's spans of the model calls the mounted session store records "
                    "whole; the store is the account read"
                ),
            ),
            Ignore(
                harbor.TRIAL,
                "agent/behavior.json",
                because="NOOA's runner's metrics computed from its trajectory, not an observation",
            ),
        ),
        attach=(
            Relate(
                frm="rollout.nooa_log",
                to="rollout.nooa_session",
                on=Join(space="nooa_session"),
                cardinality=Card.ONE,
                to_card=None,
                same_rollout=True,
                because=(
                    "a root session belongs to the trial whose agent directory holds its store; "
                    "a trial resumed into a second root session holds both"
                ),
            ),
        ),
    ),
)

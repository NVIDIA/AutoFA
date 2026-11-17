# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A harbor job: `job.log` at the folder's root and one directory per trial beneath it.

Where a host places the job is the host's `Mount`, not a location declared here.
"""

from __future__ import annotations

from evalrdl_readers.core import AnyOf, Card, Landmark, Says

JOB = Landmark(
    name="harbor_job",
    marker=AnyOf("job.log", "config.json"),
    where=("",),
    expect=Card.ONE(
        because=(
            "harbor writes one `job.log` and one `config.json` per job, at the job's root; a "
            "copy may keep only the config"
        ),
    ),
)

TRIAL = Landmark(
    name="harbor_trial",
    marker=AnyOf("result.json", "lock.json"),
    within=JOB,
    where=("*",),
    expect=Card.ANY(
        because=(
            "a trial directory is the enumeration, and a job that produced none is real. "
            "`lock.json` is in the marker for a trial that started and never wrote a "
            "result, which would otherwise vanish without a finding"
        ),
    ),
)

OWN_CONFIG = Landmark(
    name="harbor_job_config",
    marker=Says(
        "config.json",
        contains='"jobs_dir"',
        because=(
            "harbor's job config names the directory its jobs are written under, near its top; "
            "`job_name` beside it is a key many schedulers' configs share, `jobs_dir` is not"
        ),
    ),
    where=("",),
    expect=Card.ONE(because="a harbor job read on its own is named by its own config"),
)

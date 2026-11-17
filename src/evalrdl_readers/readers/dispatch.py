# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The declared order between the public readers, for folders more than one of them claims.

`Dispatch.pick` reads the first claimant and records nothing about the rest. A service with
readers of its own puts them before `PUBLIC` in a `Dispatch` of its own.
"""

from __future__ import annotations

from evalrdl_readers.core import Dispatch

from .inspect_logs.declare import READER as INSPECT
from .mounts import GYM, HARBOR, NEMO_EVALUATOR_LAUNCHER, NEMO_SKILLS
from .nooa_sessions import READER as NOOA_SESSIONS

#: The public readers, in the order they are offered a folder.
PUBLIC = (INSPECT, NEMO_EVALUATOR_LAUNCHER, NEMO_SKILLS, GYM, HARBOR, NOOA_SESSIONS)

RUNDIR = Dispatch(
    PUBLIC,
    because=(
        "the reader that needs less goes last: nemo-gym needs only the wrapper's "
        "`framework_name: nemo_gym`, nemo-skills also needs `eval-results/<task>/metrics.json`. "
        "harbor, read as a job of its own, comes after them: a launcher or a host that runs "
        "harbor holds a harbor job in its tree, and that tree is the host's run. NOOA's "
        "sessions are last: a host that runs NOOA holds its session stores in its tree"
    ),
)

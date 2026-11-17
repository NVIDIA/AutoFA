# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public readers, one package per producer.

Each package reads one producer's own files, or its own part of a file several producers wrote,
and imports no other package's reader; record formats more than one producer reads are shared
`dialects`. `mounts` composes a host with the producers it runs: where each one's files sit,
and the inputs that read a part only meaningful with two producers' knowledge. `dispatch` holds
the order between the readers a folder may be claimed by.
"""

from .dispatch import PUBLIC, RUNDIR
from .inspect_logs.declare import READER as INSPECT
from .mounts import GYM, HARBOR, NEMO_EVALUATOR_LAUNCHER, NEMO_SKILLS
from .nooa_sessions import READER as NOOA_SESSIONS

__all__ = [
    "GYM",
    "HARBOR",
    "INSPECT",
    "NEMO_EVALUATOR_LAUNCHER",
    "NEMO_SKILLS",
    "NOOA_SESSIONS",
    "PUBLIC",
    "RUNDIR",
]

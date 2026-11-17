# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every schema the store has written, oldest first."""

from __future__ import annotations

from itertools import pairwise

from .dsl import Schema, check_additive
from .v11 import V11
from .v12 import V12
from .v13 import V13
from .v14 import V14
from .v15 import V15
from .v16 import V16
from .v17 import DIALECTS, V17

#: Oldest first. Each is an additive extension of the one before, checked at import.
VERSIONS: tuple[Schema, ...] = (V11, V12, V13, V14, V15, V16, V17)

#: What new intake and every reparse writes.
CURRENT: Schema = VERSIONS[-1]

#: The request dialects `CURRENT` holds, and the request field each keeps its messages in.
CURRENT_DIALECTS: dict[str, str] = DIALECTS

#: The oldest version still readable. Nothing below this exists in the store.
EARLIEST: Schema = VERSIONS[0]

for _older, _newer in pairwise(VERSIONS):
    check_additive(_older, _newer)

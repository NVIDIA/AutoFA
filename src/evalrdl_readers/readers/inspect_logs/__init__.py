# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""inspect_ai JSON logs. Reader package: `declare` states what it claims, `translate` how it reads.

Warning: this is the repo's own synthetic test fixture (`stub-model-v1`); real inspect_ai
writes `.eval` zip logs with a richer schema, so this reader covers one fixture only.

Modules are re-exported whole, private names included, so helpers are reachable by module
attribute.
"""

from . import (
    declare,
    landmarks,
    translate,
)

for _mod in (landmarks, translate, declare):
    globals().update({_k: _v for _k, _v in vars(_mod).items() if not _k.startswith("__")})
del _mod

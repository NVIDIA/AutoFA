# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reader package: `declare` states what it claims, `translate` how it reads.

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

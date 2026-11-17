# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`Refusal`: the library's only deliberate exception; folder defects are `Finding`s instead."""

from __future__ import annotations


class Refusal(Exception):
    """A declaration that cannot be executed without guessing."""


def need_reason(because: object, what: str, instead: str) -> str:
    """Refuse an empty `because`; the shared check for every declaration's reason."""
    if not isinstance(because, str) or not because.strip():
        raise Refusal(
            f"{what}: empty `because`. {instead} "
            f'"Not looked at yet" is not a legal reason.'
        )
    return because

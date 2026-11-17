# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional accounting at the model dispatch boundary, inherited by workers."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_RESERVATION: ContextVar[Callable[[], None] | None] = ContextVar(
    "autofa_model_reservation", default=None,
)


def reserve_model_call() -> None:
    """Reserve before an SDK dispatch, including compatibility/output retries."""
    reserve = _RESERVATION.get()
    if reserve is not None:
        reserve()


@contextmanager
def model_call_budget(reserve: Callable[[], None]) -> Iterator[None]:
    token = _RESERVATION.set(reserve)
    try:
        yield
    finally:
        _RESERVATION.reset(token)

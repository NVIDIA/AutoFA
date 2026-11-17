# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The evaluator's run, read on its own; the readers of what runs inside it are mounted on it."""

from .declare import READER
from .landmarks import RUN

__all__ = ["READER", "RUN"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The nemo-evaluator envelope as a mountable reader."""

from .declare import READER, launcher_ignores
from .landmarks import ENVELOPE, RESULTS, RUN_CONFIG

__all__ = ["ENVELOPE", "READER", "RESULTS", "RUN_CONFIG", "launcher_ignores"]

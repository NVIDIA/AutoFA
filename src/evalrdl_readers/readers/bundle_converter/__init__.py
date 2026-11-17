# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A converter's bundle around a run it rewrote into an evaluator's layout, as a mountable reader."""

from .declare import READER
from .landmarks import BUNDLE

__all__ = ["BUNDLE", "READER"]

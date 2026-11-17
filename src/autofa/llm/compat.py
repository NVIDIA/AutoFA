# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM helpers exposed to sandboxed detectors (failures-scale compatible name)."""

from __future__ import annotations

from autofa.llm.hub import chat_completions_safe

__all__ = ["chat_completions_safe"]

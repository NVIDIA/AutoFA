# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors shared by normalized local and remote trace imports."""


class SourceDataError(RuntimeError):
    """A source failed retrieval, decoding, or completeness validation."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status

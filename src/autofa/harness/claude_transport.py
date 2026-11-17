# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Claude SDK transport limits."""

# The SDK defaults to 1 MiB per JSON message. Image tool results can exceed
# that even for a small PNG: base64 expands the bytes, and Claude can repeat
# the image in message.content and toolUseResult. Allow full artifact evidence
# without truncation, while retaining a finite per-message memory bound.
# This is a ceiling, not a preallocated buffer or a model context limit.
CLAUDE_MAX_BUFFER_SIZE = 64 * 1024 * 1024

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One source-level LLM signal for sampling, strategy labels and detector lint."""

import re

LLM_CALL_RE = re.compile(
    r"\bllm(?:_client)?\s*\.\s*(?:classify(?:_options)?|decide)\b"
    r"|\.\s*(?:classify(?:_options)?|decide)\s*\(|\bllm_classify\b|\bchat_completions_safe\b"
    r"|chat\s*\.\s*completions\s*\.\s*(?:create|parse|stream)"
)

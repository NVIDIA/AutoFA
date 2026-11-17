# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The key of a text: equal for two texts that read the same, so a producer's copies of one message
match where neither states an id. Versioned: a changed rule is a new version, never an edit.

The key is taken over the decoded text, never over JSON around it, so an escape (`\\u00e9`, `\\n`)
and the character it stands for agree. `text-v1` then:

- composes Unicode (NFC), so a precomposed character and its combining sequence agree; it does
  not fold compatibility forms (NFKC) or case, which change what a text says;
- ends lines with `\\n` (`\\r\\n` and `\\r` alike);
- drops a byte-order mark and the whitespace around the text, which transports add and strip;
- keeps a lone surrogate (UTF-8 `surrogatepass`) rather than failing on it.

A text that is empty after this has no key.
"""

from __future__ import annotations

import hashlib
import unicodedata

TEXT_KEY_VERSION = "text-v1"


def normalized(text: str) -> str:
    """`text` as `text-v1` compares it."""
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return t.lstrip("﻿").strip()


def text_key(text: str | None) -> str | None:
    """The `text-v1` key of `text`: its normalized form's SHA-256, or None if nothing is left."""
    if not isinstance(text, str):
        return None
    t = normalized(text)
    if not t:
        return None
    return hashlib.sha256(t.encode("utf-8", "surrogatepass")).hexdigest()

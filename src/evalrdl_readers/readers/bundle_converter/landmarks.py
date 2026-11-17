# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The converter's own files at the top of the bundle it wrote; the converted run sits beside them."""

from __future__ import annotations

from evalrdl_readers.core import AnyOf, Card, Landmark

BUNDLE = Landmark(
    name="converted_bundle",
    marker=AnyOf(
        "conversion-provenance.json",
        "conversion-report.json",
        "CONVERSION_README.md",
        "validation.json",
        "validate_conversion.py",
        "upload-attribution.json",
        "convert_*.py",
    ),
    where=("",),
    expect=Card.ONE(because="the converter writes its files at the top of the bundle"),
)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A converted bundle read on its own; a host reading the converted run mounts it with `wraps=True`."""

from __future__ import annotations

from evalrdl_readers.core import Answer, At, Card, Ignore, Input, Population, Reader, Sources

from .. import questions as Q
from ..store_schema import READER_SCHEMA
from .landmarks import BUNDLE


def bc_provenance(src: Sources):
    """`conversion-provenance.json`; the only source of `source_user`."""
    if not src.has(0):
        return
    body = src.doc(0).json() or {}
    yield Answer("run.source_user", body.get("source_user"))
    yield Answer("run.cluster", body.get("cluster"))


BC_PROVENANCE = Input(
    name="bc_provenance",
    via=bc_provenance,
    reads=[At(BUNDLE, "conversion-provenance.json")],
    answers={
        "run.source_user": Card.ANY(because="only some converters write the file"),
        "run.cluster": Card.ANY(
            because="a provenance file names a cluster only where the converted path had one in it"
        ),
    },
)

READER = Reader(
    schema=READER_SCHEMA,
    counts=Population.ATTEMPTED,
    name="bundle-converter",
    landmarks=(BUNDLE,),
    inputs=(("bundle-converter", (BC_PROVENANCE,)),),
    questions=(Q.RUN_SOURCE_USER, Q.RUN_CLUSTER),
    ignored=(
        Ignore(
            BUNDLE,
            "original/**",
            because=(
                "the converter's input, the producer's own output shipped beside the converted "
                "stream so the conversion can be re-checked; not a second account, since "
                "everything in it is in the converted stream"
            ),
        ),
        Ignore(
            BUNDLE,
            "original-archive/*.zip",
            because="the source archive the converter unpacked, whose sha256 `conversion-provenance.json` records",
        ),
        Ignore(
            BUNDLE,
            "conversion-report.json",
            because="the converter's count of what it produced; every number in it is a rollup of the converted trajectories",
        ),
        Ignore(
            BUNDLE,
            "convert_*.py",
            because="the converter program itself, shipped so the conversion is reproducible: code, not a run",
        ),
        Ignore(
            BUNDLE,
            "CONVERSION_README.md",
            because="the converter's prose account of the same conversion `conversion-report.json` counts",
        ),
        Ignore(
            BUNDLE,
            "validate_conversion.py",
            because="the converter's own checker, shipped beside the converter: code, not a run",
        ),
        Ignore(
            BUNDLE,
            "validation.json",
            because=(
                "what that checker found; every number in it is a rollup of the converted "
                "stream, and the host's count of scored rollouts against its manifest already "
                "says where the folder holds less"
            ),
        ),
        Ignore(
            BUNDLE,
            "upload-attribution.json",
            because=(
                "the uploader's authorisation to file the run under their own catalog account; "
                "it names the requester and the original source, and `runs.source_user` means "
                "the source, as `conversion-provenance.json` states"
            ),
        ),
    ),
)

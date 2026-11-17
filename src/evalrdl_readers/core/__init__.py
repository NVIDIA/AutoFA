# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""evalrdl_readers.core — a declarative reader layer for producer output folders.

A reader is a set of inputs; an input reads one kind of content and answers questions via
a translator (bytes -> answers). A landmark is a directory found by a marker file; all other
paths are rigid relative to one. Every declaration carries a reason and has no default.
Bad declarations raise `Refusal` before any file is read; folder defects become `Finding`s.

    from evalrdl_readers.core import *

    RUN = Landmark(name="run", marker="manifest.json", where=("", "artifacts"),
                   expect=Card.ONE(because="one manifest per run"))
"""

from .attach import Contribute, Lane, Link, Relate, Unlinked
from .declare import (
    AGGREGATE_OPS,
    At,
    Card,
    Compare,
    Decoder,
    Decoders,
    Fold,
    Ignore,
    Join,
    Key,
    Op,
    Order,
    Population,
    Rows,
)
from .dispatch import Dispatch
from .errors import Refusal
from .findings import LOSS, Finding, Kind
from .input import Answer, Doc, Given, Input, Sources, Translator, Unreadable
from .landmarks import ROOT, Landmark, Site, resolve_landmarks
from .mount import Mount
from .patterns import AnyOf, Says
from .questions import (
    DEPTH,
    LOWEST_FIRST,
    PARENT,
    TABLES,
    Grain,
    Merge,
    Question,
)
from .reader import Reader
from .result import Edge, ReadResult, Row
from .tree import Entry, Index, LocalTree, Tree

__version__ = "0.1.0"

__all__ = [
    # declarations
    "Landmark",
    "AnyOf",
    "Says",
    "At",
    "Input",
    "Card",
    "Population",
    "Rows",
    "Order",
    "Fold",
    "Op",
    "Compare",
    "Ignore",
    "Question",
    "Grain",
    "Merge",
    "Contribute",
    "Lane",
    "Link",
    "Unlinked",
    "Relate",
    "Join",
    "Key",
    "Decoders",
    "Decoder",
    "Reader",
    "Dispatch",
    "Mount",
    # translator side
    "Answer",
    "Unreadable",
    "Sources",
    "Doc",
    "Translator",
    # output
    "ReadResult",
    "Row",
    "Edge",
    "Given",
    "Finding",
    "Kind",
    "LOSS",
    "Site",
    # refusal
    "Refusal",
    # trees
    "Tree",
    "LocalTree",
    "Index",
    "Entry",
    # the fixed ladders and lists
    "TABLES",
    "DEPTH",
    "PARENT",
    "LOWEST_FIRST",
    "AGGREGATE_OPS",
    "ROOT",
    "resolve_landmarks",
    "__version__",
]

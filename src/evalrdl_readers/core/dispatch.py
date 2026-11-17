# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`Dispatch`: a declared order between readers that may claim one folder."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import Refusal
from .findings import Finding, Kind, sort_findings
from .reader import Reader, _as_tree
from .result import ReadResult
from .tree import Tree


@dataclass(frozen=True)
class Dispatch:
    """A declared order between readers that may claim one folder; the first to claim wins.

    The reader that needs less goes last.
    """

    readers: tuple[Reader, ...]
    """The readers, in the order they are offered a folder."""
    because: str
    """Why this order: what each earlier reader excludes."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "readers", tuple(self.readers))
        if len(self.readers) < 2:
            raise Refusal(
                "Dispatch: an order of one settles nothing. One reader needs no dispatch."
            )
        for r in self.readers:
            if not isinstance(r, Reader):
                raise Refusal(f"Dispatch: expected Reader, got {type(r).__name__}")
        names = [r.name for r in self.readers]
        if len(set(names)) != len(names):
            raise Refusal(f"Dispatch: two readers named {sorted(names)}")
        if not isinstance(self.because, str) or not self.because.strip():
            raise Refusal(
                "Dispatch: empty `because`. The reader that needs less goes last, and "
                "the reason has to name what the earlier reader excludes."
            )

    def pick(self, tree: Tree | str | os.PathLike[str]) -> Reader | None:
        """The first reader that claims this folder, in the declared order."""
        t = _as_tree(tree)
        for r in self.readers:
            claimed, _why = r.detect(t)
            if claimed:
                return r
        return None

    def read(self, tree: Tree | str | os.PathLike[str]) -> ReadResult:
        """The first claimant's read; unclaimed, a result naming why each reader declined."""
        t = _as_tree(tree)
        r = self.pick(t)
        if r is None:
            out = ReadResult(
                reader="<none>", tree=getattr(t, "name", str(tree)), claimed=False
            )
            for reader_ in self.readers:
                _claimed, why = reader_.detect(t)
                for w in why:
                    out.findings.append(
                        Finding(Kind.MISSING, f"reader:{reader_.name}", out.tree, w)
                    )
            out.findings = sort_findings(out.findings)
            return out
        return r.read(t)

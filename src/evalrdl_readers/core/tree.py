# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `Tree` interface over a run's bytes, and the `Index` built on it.

Only `paths()` + `read()`: no recursive glob, so `**/name` is not expressible.
"""

from __future__ import annotations

import os
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol

from .patterns import is_literal

if TYPE_CHECKING:
    from .patterns import Says


@dataclass(frozen=True, order=True)
class Entry:
    """One file in the tree. Always a file, always tree-relative, always posix."""

    path: str
    size: int | None = None


def _by_path(e: Entry) -> str:
    return e.path


class Tree(Protocol):
    """What a reader is pointed at. A tree may also have `open(path)` returning a binary
    stream; `Doc.jsonl()` then reads line by line instead of whole.

    Attributes:
        name: The folder's name, as results name it.
    """

    name: str

    def paths(self) -> list[Entry]:
        """Every file, as tree-relative posix paths with sizes."""
        ...

    def read(self, path: str, limit: int | None = None) -> bytes:
        """One file's bytes, or its first `limit` of them."""
        ...


#: Records run to megabytes; a larger buffer joins fewer chunks per line.
_STREAM_BUFFER = 1 << 20


class LocalTree:
    """A directory on disk; a path given to `read` is one.

    Arguments:
        root: The directory.
        name: The folder's name in results; by default the directory's.
    """

    def __init__(self, root: str | os.PathLike[str], name: str | None = None):
        self.root = Path(root)
        self.name = name or self.root.name
        self._paths: list[Entry] | None = None

    def paths(self) -> list[Entry]:
        """Every file under the directory, sorted, listed once."""
        if self._paths is None:
            top = str(self.root)
            out: list[Entry] = []
            for dirpath, _dirs, files in os.walk(top):
                rel = Path(dirpath).relative_to(top).as_posix()
                prefix = "" if rel == "." else f"{rel}/"
                for f in files:
                    try:
                        size = os.stat(os.path.join(dirpath, f)).st_size
                    except OSError:
                        size = None
                    out.append(Entry(prefix + f, size))
            out.sort(key=_by_path)
            self._paths = out
        return self._paths

    def read(self, path: str, limit: int | None = None) -> bytes:
        """One file's bytes, or its first `limit` of them."""
        with (self.root / path).open("rb") as f:
            return f.read() if limit is None else f.read(limit)

    def open(self, path: str) -> BinaryIO:
        """The file as a binary stream; `Doc.jsonl()` uses it, where a tree has it, to
        read line by line instead of whole."""
        return (self.root / path).open("rb", buffering=_STREAM_BUFFER)

    def __getstate__(self) -> dict:
        # A reader child opens files by path; the listing stays with the parent.
        return {**self.__dict__, "_paths": None}

    def __repr__(self) -> str:
        return f"<LocalTree {self.root}>"


def _parent(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _join(d: str, name: str) -> str:
    return f"{d}/{name}" if d else name


class Index:
    """Directory + file index of one tree, built once from the flat path list.

    Patterns are walked segment by segment from their base, so a lookup costs the
    directories it can reach, not every directory in the tree.
    """

    def __init__(self, tree: Tree):
        self.tree = tree
        self.files: list[Entry] = list(tree.paths())
        self.by_dir: dict[str, list[Entry]] = {}
        self.dirs: set[str] = {""}
        for e in self.files:
            parent = _parent(e.path)
            self.by_dir.setdefault(parent, []).append(e)
            d = parent
            while d not in self.dirs:
                self.dirs.add(d)
                d = _parent(d)
        for entries in self.by_dir.values():
            entries.sort(key=_by_path)
        self._file: dict[str, Entry] = {e.path: e for e in self.files}
        self._subdirs: dict[str, list[tuple[str, str]]] = {}
        for d in self.dirs:
            if d:
                self._subdirs.setdefault(_parent(d), []).append((d.rsplit("/", 1)[-1], d))
        self._sorted = sorted(self.files, key=_by_path)
        self._sorted_paths = [e.path for e in self._sorted]

    def dirs_at(self, base: str, pat: tuple[str, ...]) -> list[str]:
        """Directories matching a rigid relative pattern below `base`. `()` means `base`."""
        if base not in self.dirs:
            return []
        if not pat:
            return [base]
        level = [base]
        for seg in pat:
            nxt: list[str] = []
            if is_literal(seg):
                for d in level:
                    child = _join(d, seg)
                    if child in self.dirs:
                        nxt.append(child)
            else:
                for d in level:
                    nxt.extend(c for name, c in self._subdirs.get(d, ()) if fnmatchcase(name, seg))
            if not nxt:
                return []
            level = nxt
        return sorted(level)

    def files_at(self, base: str, pat: tuple[str, ...]) -> list[Entry]:
        """Files matching a rigid relative pattern below `base`."""
        if not pat:
            return []
        if pat[-1] == "**":
            out = []
            for d in self.dirs_at(base, pat[:-1]):
                out.extend(self._files_under(d))
            return sorted(out, key=_by_path)
        parent_pat, name_pat = pat[:-1], pat[-1]
        dirs = [base] if not parent_pat else self.dirs_at(base, parent_pat)
        out = []
        if is_literal(name_pat):
            for d in dirs:
                e = self._file.get(_join(d, name_pat))
                if e is not None:
                    out.append(e)
        else:
            for d in dirs:
                for e in self.by_dir.get(d, ()):
                    if fnmatchcase(e.path.rsplit("/", 1)[-1], name_pat):
                        out.append(e)
        return sorted(out, key=_by_path)

    def _files_under(self, d: str) -> list[Entry]:
        """Every file below directory `d`, at any depth."""
        if not d:
            return list(self._sorted)
        prefix = f"{d}/"
        paths = self._sorted_paths
        i = bisect_left(paths, prefix)
        j = i
        while j < len(paths) and paths[j].startswith(prefix):
            j += 1
        return self._sorted[i:j]

    def marker_says(self, d: str, says: Says) -> tuple[bool, str | None]:
        """Return (matched, complaint): does any named marker in `d` carry the literal in
        its head bytes? An empty/unreadable marker is a complaint, never a silent non-match.
        """
        from .patterns import HEAD_BYTES

        complaint: str | None = None
        for e in self.by_dir.get(d, ()):
            base = e.path.rsplit("/", 1)[-1]
            if not any(fnmatchcase(base, n) for n in says.names):
                continue
            try:
                head = self.tree.read(e.path, limit=HEAD_BYTES)
            except OSError as exc:
                complaint = complaint or f"{e.path}: unreadable ({exc}) — {says.because}"
                continue
            if not head:
                complaint = complaint or (
                    f"{e.path}: is empty, so `contains={says.contains!r}` cannot be "
                    f"judged. An empty marker is a finding, not a non-match — {says.because}"
                )
                continue
            if any(c.encode() in head for c in says.contains):
                return True, None
        return False, complaint

    def has_marker(self, d: str, names: Iterable[str]) -> bool:
        entries = self.by_dir.get(d, ())
        for n in names:
            if is_literal(n):
                if _join(d, n) in self._file:
                    return True
            elif any(fnmatchcase(e.path.rsplit("/", 1)[-1], n) for e in entries):
                return True
        return False

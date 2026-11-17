# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval log source discovery (parquet, NeMo Gym dirs, JSONL)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SourceKind = Literal["parquet", "gym_dir", "jsonl", "evalrdl"]


@dataclass(frozen=True)
class EvalSource:
    """One selectable eval log unit."""

    kind: SourceKind
    path: Path
    """Directory (gym_dir) or file (parquet / jsonl)."""
    hint: str = ""
    files: tuple[Path, ...] = field(default_factory=tuple)
    """Concrete input files (parquets or jsonls)."""


def is_gym_dir(path: Path) -> bool:
    """True if ``path`` looks like a NeMo Gym / terminal-bench log folder."""
    path = Path(path)
    if not path.is_dir():
        return False
    if (path / "eval_metadata.json").is_file():
        return True
    jsonls = list(path.glob("*.jsonl"))
    if not jsonls:
        return False
    # Prefer explicit naming conventions from NeMo Gym dumps.
    names = [p.name.lower() for p in jsonls]
    if any("parsed_trajectories" in n for n in names):
        return True
    if any("rollout" in n for n in names):
        return True
    return False


def recognized_log_folder(path: Path) -> str | None:
    """Recognize a run or its single-job envelope without scanning whole logs.

    Explicit framework declarations are sufficient even for empty/partial runs.
    Undeclared stream folders need a coherent layout; mixed download directories
    and containers of several runs must not inherit one child's recognition.
    """
    import os
    import re

    path = Path(path)
    if not path.is_dir():
        return None

    def prefix(file: Path, limit: int = 65536) -> str:
        try:
            if file.is_symlink() or not file.is_file():
                return ""
            with file.open("rb") as stream:
                return stream.read(limit).decode("utf-8", errors="replace")
        except OSError:
            return ""

    def listing(folder: Path) -> list[Path]:
        try:
            with os.scandir(folder) as iterator:
                return [Path(entry.path) for entry in iterator
                        if not entry.name.startswith(".") and not entry.is_symlink()]
        except OSError:
            return []

    def gym_run(folder: Path) -> bool:
        # These are also the vendored reader's run landmarks. Config can be the
        # only witness when streams are empty; unrelated output assets are fine.
        config = "\n".join(prefix(folder / name) for name in ("run_config.yml", "metadata.yaml"))
        if re.search(r'["\']?framework_name["\']?\s*:\s*["\']?nemo_gym\b'
                     r'|["\']nemo_gym["\']\s*:\s*\{', config):
            return True
        entries = listing(folder)
        streams = sorted(p for p in entries if p.is_file() and p.suffix == ".jsonl")
        if not streams:
            return False
        stems = {p.stem for p in streams}
        directories = {"artifacts", "results", "logs", "nemo_gym_logs", "model_calls", "harbor"}
        suffixes = {".json", ".jsonl", ".yaml", ".yml", ".log", ".out", ".err", ".txt", ".md"}
        if any(p.name not in directories | stems if p.is_dir()
               else p.suffix.lower() not in suffixes for p in entries):
            return False
        # Stream signatures are enough without optional eval_metadata.json.
        # Sample across large shard sets instead of rejecting folders by size.
        sample = streams if len(streams) <= 32 else [
            streams[i * (len(streams) - 1) // 31] for i in range(32)]
        recognizable = []
        for stream in sample:
            content = prefix(stream)
            if re.search(r'"(?:_ng_task_index|responses_create_params)"\s*:', content) or (
                "parsed_trajectories" in stream.name and '"steps"' in content
            ):
                recognizable.append(stream)
        if not recognizable:
            return False
        recognized_stems = {p.stem for p in recognizable}
        return all(p in recognizable or p.stat().st_size == 0 or p.stem == "rollout_verdicts"
                   or any(p.stem.startswith(stem + "_") for stem in recognized_stems)
                   for p in sample)

    gym_label = "NeMo Gym log folder recognized — use this folder?"
    # Gym can embed Harbor jobs; prefer its outer run envelope.
    if gym_run(path):
        return gym_label
    # A Harbor config identifies even a job with zero/unfinished trials. Read its
    # signature without parsing a potentially truncated, large config document.
    if re.search(r'"jobs_dir"\s*:\s*"', prefix(path / "config.json")):
        return "Harbor log folder recognized — use this folder?"

    entries = listing(path)
    children = [p for p in entries if p.is_dir()]
    envelopes = [path / name for name in ("artifacts", "results")]
    envelopes.extend(child / "artifacts" for child in children)
    matches = [folder for folder in envelopes if folder.is_dir() and gym_run(folder)]
    if len(matches) != 1:
        return None
    # Recognize <job>/artifacts, but never a mixed parent simply because one of
    # its children is a run. Auxiliary files and scheduler logs are allowed.
    branch = matches[0].relative_to(path).parts[0]
    if all(p.name in {branch, "logs", "model_calls"} for p in children) and not any(
        p.suffix == ".jsonl" for p in entries if p.is_file()
    ):
        return gym_label
    return None


def discover_parquets(root: Path) -> list[Path]:
    """Back-compat: all ``*.parquet`` under ``root`` (or the file itself)."""
    root = Path(root)
    if root.is_file() and root.suffix == ".parquet":
        return [root]
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def discover_eval_sources(root: Path) -> list[EvalSource]:
    """Discover parquet / gym / jsonl sources under ``root``.

    Preference:
    - If ``root`` is a single parquet → one parquet source
    - If ``root`` is a gym dir (metadata or gym-named jsonl) → one gym_dir source
      (even if stray parquets exist)
    - If ``root`` is a jsonl file → one jsonl source
    - Else: all parquets under the tree; if none, gym subdirs / loose jsonls
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return []

    from autofa.data.local_import import detect_reader, reference_for_import

    if root.is_dir() and (
        (root / "source.json").is_file()
        or ((root / "dataset.parquet").is_file() and (root / "manifest.json").is_file())
    ):
        dataset = root / "dataset.parquet"
        if reference_for_import(dataset):
            return [EvalSource(kind="parquet", path=dataset, hint=root.name, files=(dataset,))]
    if root.is_dir() and (root / "rollouts.parquet").is_file():
        import pyarrow.parquet as pq

        columns = set(pq.ParquetFile(root / "rollouts.parquet").schema_arrow.names)
        if {"run_id", "task_id", "repeat"} <= columns:
            raise ValueError(
                "This folder contains normalized run tables. Select the original harness "
                "run folder or its managed dataset.parquet, rather than individual tables."
            )

    if reader := detect_reader(root):
        return [EvalSource(kind="evalrdl", path=root, hint=f"{reader} · {root.name}")]

    if root.is_file():
        if root.suffix == ".parquet":
            return [EvalSource(kind="parquet", path=root, hint=root.name, files=(root,))]
        if root.suffix == ".jsonl":
            return [
                EvalSource(
                    kind="jsonl",
                    path=root,
                    hint=f"JSONL · {root.name}",
                    files=(root,),
                )
            ]
        return []

    # Directory: gym dump takes precedence as a single unit.
    if is_gym_dir(root):
        files = _gym_input_files(root)
        return [
            EvalSource(
                kind="gym_dir",
                path=root,
                hint=f"NeMo Gym · {root.name}",
                files=tuple(files),
            )
        ]

    # Recognize run boundaries before collecting loose files. A Harbor run may
    # contain artifact Parquets, and a mixed selection must not hide its runs.
    declared = []
    remaining = []
    for child in sorted(root.iterdir()):
        if child.is_symlink() or child.name.startswith("."):
            continue
        if child.is_dir():
            found = discover_eval_sources(child)
            declared.extend(s for s in found if s.kind != "parquet")
            remaining.extend(s for s in found if s.kind == "parquet")
        elif child.suffix == ".parquet":
            remaining.append(EvalSource(kind="parquet", path=child, hint=child.name,
                                        files=(child,)))
    if declared or remaining:
        loose = [EvalSource(kind="jsonl", path=p, hint=f"JSONL · {p.name}", files=(p,))
                 for p in sorted(root.glob("*.jsonl"))]
        return declared + remaining + loose

    # No parquet: look for nested gym dirs, then loose jsonl.
    out: list[EvalSource] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if is_gym_dir(child):
            files = _gym_input_files(child)
            out.append(
                EvalSource(
                    kind="gym_dir",
                    path=child,
                    hint=f"NeMo Gym · {child.name}",
                    files=tuple(files),
                )
            )
    if out:
        return out

    for jl in sorted(root.rglob("*.jsonl")):
        out.append(
            EvalSource(
                kind="jsonl",
                path=jl,
                hint=f"JSONL · {jl.name}",
                files=(jl,),
            )
        )
    return out


def _gym_input_files(dir_path: Path) -> list[Path]:
    """Prefer parsed trajectories; else rollouts; else any jsonl."""
    dir_path = Path(dir_path)
    parsed = sorted(dir_path.glob("*parsed_trajectories*.jsonl"))
    if parsed:
        return parsed
    rollouts = sorted(
        p
        for p in dir_path.glob("*.jsonl")
        if "rollout" in p.name.lower()
    )
    if rollouts:
        return rollouts
    return sorted(dir_path.glob("*.jsonl"))

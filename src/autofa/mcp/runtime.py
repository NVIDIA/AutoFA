# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MCP runtime state."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autofa.config import Config, RunConfig, load_config
from autofa.core.library import FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.util.atomic import atomic_write_json
from autofa.util.cancellation import cancellation_guard


@dataclass
class Runtime:
    dataset: Dataset
    library: FunctionLibrary
    run: RunConfig
    config: Config
    scores_history: list[dict[str, Any]] = field(default_factory=list)
    # OpenAI-compatible client for detectors / llm_classify (legacy attribute name).
    hub_llm: Any | None = None
    # Startup probe result, shared with the discovery agent via dataset_info.
    llm_readiness: dict[str, Any] | None = None
    _scores_history_lock: Any = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False,
    )
    _library_lock: Any = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False,
    )

    @property
    def helpers(self):
        return self.library.helpers

    def working_library_path(self) -> Path:
        return self.run.run_dir() / "library.json"

    def persist_library(
        self, *, repairs: list[dict] | None = None, library: FunctionLibrary | None = None,
    ) -> Path:
        """Persist a validated library, publishing staged imports only after success."""
        from autofa.session.exploration import initialize_run_exploration
        from autofa.session.review_evidence import sync_review_history

        # A failed submission must roll back before another tool can serialize
        # the library. Always acquire cancellation before the library lock.
        with cancellation_guard(), self._library_lock:
            path = self.working_library_path()
            candidate = self.library if library is None else library
            candidate.model_id = self.run.library_model_id()
            candidate.benchmark = self.run.benchmark_label()
            initialize_run_exploration(self.run)
            candidate.exploration_state = {
                "version": 1, "base_seed": self.run.exploration_base_seed,
                "sequence_index": self.run.exploration_sequence_index,
            }
            sync_review_history(candidate, self.run.run_dir(), repairs=repairs,
                                persist=lambda: candidate.save(path))
            if library is not None:
                self.library = candidate
        return path

    def persist_scores_history(self) -> Path:
        # Writers hold this reentrant lock across history selection/mutation
        # and persistence, so serialization cannot race with another writer.
        with cancellation_guard(), self._scores_history_lock:
            path = self.run.run_dir() / "scores_history.json"
            atomic_write_json(path, self.scores_history)
        return path

    def ensure_hub_llm(self) -> Any:
        if self.hub_llm is not None:
            return self.hub_llm
        from autofa.llm.endpoints import endpoint_for_config
        from autofa.llm.hub import HubLLM, make_hub_client

        endpoint = endpoint_for_config(self.config)
        client = make_hub_client(
            base_url=endpoint.base_url,
            api_key_env=endpoint.api_key_env,
        )
        self.hub_llm = HubLLM(client, model=endpoint.model,
                             request_options=endpoint.request_options, api=endpoint.api)
        return self.hub_llm


_RUNTIME: Runtime | None = None


def set_runtime(rt: Runtime) -> None:
    global _RUNTIME
    _RUNTIME = rt


def get_runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("AutoFA runtime not configured")
    return _RUNTIME


def ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def err(msg: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": msg, **extra}


def load_runtime_from_run(run: RunConfig, cfg: Config | None = None) -> Runtime:
    from autofa.objectives import config_for_run
    from autofa.session.classifier_profiles import apply_run_classifier
    from autofa.session.run_capacity import apply_run_capacity

    cfg = config_for_run(apply_run_classifier(cfg or load_config(), run), run)
    cfg = apply_run_capacity(cfg, run)
    label = run.benchmark_label()
    library_model = run.library_model_id()
    ds = Dataset(
        parquet_paths=[Path(p) for p in run.parquet_paths],
        model_id=None if run.model_id in {"*", "ALL", ""} else run.model_id,
        benchmarks=run.resolved_benchmarks(),
    )
    lib_path = run.run_dir() / "library.json"
    if lib_path.exists():
        library = FunctionLibrary.load(lib_path)
        library.model_id = library_model
        library.benchmark = label
    elif run.seed_library_path and Path(run.seed_library_path).exists():
        from autofa.config import runs_root
        from autofa.session.review_evidence import import_seed_review_history

        library = FunctionLibrary.load(Path(run.seed_library_path))
        import_seed_review_history(library, Path(run.seed_library_path), runs_root=runs_root())
        library.model_id = library_model
        library.benchmark = label
    else:
        library = FunctionLibrary(model_id=library_model, benchmark=label)

    hist_path = run.run_dir() / "scores_history.json"
    history: list[dict[str, Any]] = []
    if hist_path.exists():
        history = json.loads(hist_path.read_text(encoding="utf-8"))

    from autofa.session.review_evidence import sync_review_history

    sync_review_history(library, run.run_dir())
    rt = Runtime(dataset=ds, library=library, run=run, config=cfg, scores_history=history)
    set_runtime(rt)
    return rt

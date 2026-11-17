# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and explicitly execute isolated, unassisted baseline/updated AutoFA arms.

Preparation never imports either runtime or calls a model. Execution is opt-in and
uses native AutoFA; it supplies no detectors, controls, examples or repair code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def tree_identity(root: Path, *, code: bool = False) -> str:
    digest = hashlib.sha256()
    paths = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and (not code or p.suffix in {".py", ".toml", ".md"}))
    if not paths:
        raise ValueError(f"No input files under {root}")
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def reject_literal_credentials(value: object) -> None:
    """Keep reproducibility records credential-free; use native *_key_env routing.

    Mirror profile-preview sensitivity keys without importing either AutoFA runtime:
    preparation remains usable with the standard library alone.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = "".join(c for c in str(key).lower() if c.isalnum())
            sensitive = normalized in {
                "authorization", "proxyauthorization", "authentication", "auth",
                "token", "tokens", "secret", "password", "passwd", "credential",
                "credentials", "cookie", "cookies", "apikey", "key", "privatekey",
                "headers", "extraheaders",
            } or normalized.endswith(("token", "apikey", "password", "secret", "headers"))
            if sensitive and child not in (None, "", {}, []):
                raise ValueError("Literal credentials or headers cannot be saved in evaluation "
                                 "settings; configure API key environment-variable names instead")
            reject_literal_credentials(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_literal_credentials(child)


def config_fields(checkout: Path) -> set[str]:
    """Read the checkout's Config field names statically, without importing its runtime."""
    path = checkout / "src" / "autofa" / "config.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            return {item.target.id for item in node.body
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                    and "ClassVar" not in ast.unparse(item.annotation)}
    raise ValueError(f"No Config class in {path}")


def prepare(output: Path, *, baseline: Path, updated: Path, source: Path,
            settings: dict, seeds: list[int], timeout_seconds: int, mode: str) -> Path:
    """Pin raw inputs, shared explicit settings and code; create clean per-arm homes."""
    reject_literal_credentials(settings)
    if timeout_seconds <= 0 or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Use a positive wall-clock limit and distinct exploration seeds")
    if (settings.get("skill_paths") or settings.get("skills_dir")
            or settings.get("preflight_detectors")):
        raise ValueError("Unassisted arms cannot receive external skills or preflight findings")
    if mode == "controlled" and not all(key in settings for key in (
        "objective_axes", "objective_rank", "objective_weights",
    )):
        raise ValueError("Controlled comparisons require all three explicit objective fields")
    if not isinstance(settings.get("classifier_profiles"), list):
        raise ValueError("Explicitly supply the same allowed classifier_profiles (possibly [])")
    if not isinstance(settings.get("max_rounds"), int) or settings["max_rounds"] <= 0:
        raise ValueError("Explicitly supply a positive max_rounds")
    # No target-driven automatic extension can escape the declared comparison budget.
    settings = {**settings, "min_failure_modes": 0, "discovery_wait_for_plateau": False,
                "skill_paths": [], "analysis_outputs": ["agent"], "preflight_detectors": []}
    source = source.resolve()
    checkouts = {"baseline": baseline.resolve(), "updated": updated.resolve()}
    # Both arms build Config(**settings); an unknown key would fail only after launch.
    fields = set.intersection(*(config_fields(path) for path in checkouts.values()))
    if unknown := sorted({*settings, "exploration_seed"} - fields):
        raise ValueError("Settings are not Config fields in both checkouts: "
                         + ", ".join(unknown))
    identities = {name: tree_identity(path / "src" / "autofa", code=True)
                  for name, path in checkouts.items()}
    source_hash = tree_identity(source)
    output.mkdir(parents=True, exist_ok=False)
    arms = []
    for seed in seeds:
        for name, checkout in checkouts.items():
            directory = output / f"{name}-{seed}"
            directory.mkdir()
            configured = {**settings, "exploration_seed": seed}
            (directory / "settings.json").write_text(json.dumps(configured, indent=2) + "\n")
            arms.append({"name": name, "seed": seed, "checkout": str(checkout),
                         "code_identity": identities[name], "directory": str(directory.resolve())})
    manifest = {"version": 1, "scope": "unassisted_raw_data", "mode": mode,
                "source": str(source), "source_identity": source_hash,
                "settings": settings, "timeout_seconds": timeout_seconds, "arms": arms,
                "interpretation": "Requires live native discovery and independent evaluation; "
                                  "preparation or synthetic execution establishes no recall gain."}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


_ARM_PROGRAM = '''
import json, sys
from dataclasses import asdict
from pathlib import Path
from autofa.config import Config
from autofa.cli import run_autofa
from autofa.run_config import AutoFAConfig, AutoFAExecution, SourceSpec
arm, source = Path(sys.argv[1]), Path(sys.argv[2])
cfg = Config(**json.loads((arm / 'settings.json').read_text()))
(arm / 'effective-settings.json').write_text(json.dumps(asdict(cfg), indent=2))
run = run_autofa(AutoFAConfig(cfg, SourceSpec(local_folder=source),
    AutoFAExecution(mode='batch', publish=False, require_llm_detectors=True)))
(arm / 'native-result.json').write_text(json.dumps({'run_id': run.run_id,
    'run_dir': str(run.run_dir()), 'scoring': 'native', 'harness_candidate_injections': 0,
    'harness_case_leads': 0, 'harness_repairs': 0,
    'external_interventions': 'not_audited'}, indent=2))
'''


def execute(path: Path, *, python: str, arm_name: str, seed: int) -> dict:
    """Launch one explicitly selected arm; interrupts rather than silently reruns on timeout."""
    manifest = json.loads(path.read_text())
    arm = next(a for a in manifest["arms"] if a["name"] == arm_name and a["seed"] == seed)
    source, directory, checkout = (Path(manifest["source"]), Path(arm["directory"]),
                                   Path(arm["checkout"]))
    if tree_identity(source) != manifest["source_identity"]:
        raise ValueError("Source files changed; prepare a new comparison")
    if tree_identity(checkout / "src" / "autofa", code=True) != arm["code_identity"]:
        raise ValueError("Runtime source changed; prepare a new comparison")
    settings = json.loads((directory / "settings.json").read_text())
    if settings != {**manifest["settings"], "exploration_seed": seed}:
        raise ValueError("Arm settings changed; prepare a new comparison")
    # Existing work cannot be silently rerolled for a better outcome.
    receipt_path = directory / "execution.json"
    with receipt_path.open("x") as stream:
        json.dump({"status": "starting", "python": python}, stream)
    env = {**os.environ, "AUTOFA_HOME": str(directory / "home"),
           "AUTOFA_CONFIG": str(directory / "unused-config.toml"),
           "PYTHONPATH": str(checkout / "src")}
    status = "completed"
    try:
        log_stream = (directory / "native.log").open("wb")
    except OSError:
        # A local output failure also precedes launching any model work.
        receipt_path.unlink()
        raise
    with log_stream as log:
        try:
            process = subprocess.Popen([python, "-c", _ARM_PROGRAM, str(directory), str(source)],
                                       cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        except (OSError, ValueError):
            # No child started: permit a corrected executable without rerolling
            # an experiment that actually executed.
            receipt_path.unlink()
            raise
        try:
            returncode = process.wait(timeout=manifest["timeout_seconds"])
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            status = "timed_out" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted"
            os.killpg(process.pid, signal.SIGINT)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    unchanged = (tree_identity(source) == manifest["source_identity"]
                 and tree_identity(checkout / "src" / "autofa", code=True) == arm["code_identity"])
    result_error = None
    if status == "completed" and returncode == 0:
        try:
            native = json.loads((directory / "native-result.json").read_text())
            if (not isinstance(native, dict) or not native.get("run_id")
                    or not native.get("run_dir") or native.get("scoring") != "native"):
                raise ValueError("Native result has no valid run identity or scoring origin")
        except (OSError, ValueError) as exc:
            status = "missing_result"
            result_error = str(exc)
    result = {"status": ("inputs_changed" if not unchanged else
                         status if returncode == 0 or status != "completed" else "failed"),
              "returncode": returncode, "python": python, "harness_candidate_injections": 0,
              "external_interventions": "not_audited",
              "instruction": "Inspect native traces, versions, final coverage, unknowns, costs "
                             "and external interventions before claiming autonomous success."}
    if result_error is not None:
        result["result_error"] = result_error
    receipt_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="No inference: pin a new comparison")
    for name in ("output", "baseline", "updated", "source", "settings"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--seeds", nargs="+", type=int, required=True)
    prep.add_argument("--timeout-seconds", type=int, required=True)
    prep.add_argument("--mode", choices=["controlled", "default-experience"], required=True)
    run = commands.add_parser("execute", help="Launch one bounded arm; incurs model usage")
    run.add_argument("manifest", type=Path)
    run.add_argument("--arm", choices=["baseline", "updated"], required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.output, baseline=args.baseline, updated=args.updated, source=args.source,
                      settings=json.loads(args.settings.read_text()), seeds=args.seeds,
                      timeout_seconds=args.timeout_seconds, mode=args.mode))
        return 0
    result = execute(args.manifest, python=args.python, arm_name=args.arm, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

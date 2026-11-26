# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User and run configuration for AutoFA."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import tomllib
import urllib.parse
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tomlkit

from autofa import defaults
from autofa.behavior_policy import validate_behavior_strictness
from autofa.data.reference_data import (
    make_reference_data,
    normalize_reference_data,
    summarize_source_harness,
)
from autofa.defaults import default_field, harness_default, setting_value


def global_env_path() -> Path:
    """Return the shared dotenv path without creating the profile directory."""
    return Path(os.environ.get("AUTOFA_HOME", "~/.autofa")).expanduser().resolve() / ".env"


def load_dotenv(path: Path | None = None) -> Path | None:
    """Load dotenv values, retaining environment > nearest project > global precedence.

    Project files (cwd and parents) supplement each other, then the shared
    ``$AUTOFA_HOME/.env`` or ``~/.autofa/.env`` fills remaining keys. An explicit
    ``path`` loads only that file. Returns the first existing file, if any.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path).expanduser())
    else:
        here = Path.cwd().resolve()
        for parent in [here, *here.parents]:
            candidates.append(parent / ".env")
            if (parent / "pyproject.toml").exists() and (parent / "src" / "autofa").exists():
                break
        candidates.append(global_env_path())

    seen: set[Path] = set()
    first_loaded: Path | None = None
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        if first_loaded is None:
            first_loaded = resolved
        for raw in resolved.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith(("export ", "export\t")):
                line = line[6:].lstrip()
            key, _, val = line.partition("=")
            key = key.strip()
            if not key or key in os.environ:
                continue
            val = val.strip()
            if val.startswith(("\"", "'")):
                closing = 1
                while closing < len(val) and val[closing] != val[0]:
                    closing += 2 if val[closing] == "\\" else 1
                if closing < len(val) and (
                    not val[closing + 1:].strip() or val[closing + 1:].lstrip().startswith("#")
                ):
                    val = val[1:closing]
            else:
                val = re.split(r"\s+#", val, maxsplit=1)[0].rstrip()
            os.environ[key] = val
    return first_loaded


def autofa_home() -> Path:
    raw = os.environ.get("AUTOFA_HOME", "~/.autofa")
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def library_root() -> Path:
    root = autofa_home() / "library"
    root.mkdir(parents=True, exist_ok=True)
    return root


def runs_root() -> Path:
    root = autofa_home() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def remote_library_cache_root() -> Path:
    """Cache root for libraries downloaded from the remote service."""
    root = autofa_home() / "cache" / "remote-libraries"
    root.mkdir(parents=True, exist_ok=True)
    return root


def validate_analysis_outputs(
    value: list[str], preflight_detectors: list[str] | None = None,
) -> list[str]:
    """Require at least one analysis or preflight check for new runs."""
    from autofa.preflight.registry import validate_detectors

    enabled = validate_detectors([] if preflight_detectors is None else preflight_detectors)
    if (not isinstance(value, list) or (not value and not enabled)
            or any(item not in ("efficiency", "agent") for item in value)):
        raise ValueError("analysis_outputs must contain efficiency, agent, or both "
                         "unless a preflight check is enabled")
    return list(dict.fromkeys(value))


def validate_min_failure_modes(value: int) -> int:
    """Validate without coercing booleans, floats, or numeric strings."""
    if type(value) is not int or value < 0:
        raise ValueError("min_failure_modes must be a non-negative integer (0 disables it)")
    return value


def validate_discovery_wait_for_plateau(value: bool) -> bool:
    """Require an explicit boolean for the optional discovery stopping condition."""
    if type(value) is not bool:
        raise ValueError("discovery_wait_for_plateau must be a boolean")
    return value


def validate_discovery_terminate_if_stalled(value: bool) -> bool:
    """Require an explicit boolean for automatic stalled-discovery termination."""
    if type(value) is not bool:
        raise ValueError("discovery_terminate_if_stalled must be a boolean")
    return value


def validate_discovery_stall_rounds(value: int) -> int:
    """Count complete recovery rounds after the planned discovery budget."""
    if type(value) is not int or value < 1:
        raise ValueError("discovery_stall_rounds must be a positive integer")
    return value


def validate_objective_rank(value: str) -> str:
    """Rank objectives by the weighted geometric mean, the only supported aggregation."""
    if value != "geometric_mean":
        raise ValueError(f"objective_rank must be 'geometric_mean', not {value!r}")
    return value


def validate_taxonomy_first_pass(value: str) -> str:
    """Validate the taxonomy first-pass method."""
    if not isinstance(value, str) or value not in {"llm", "agentic"}:
        raise ValueError("taxonomy_first_pass must be llm or agentic")
    return value


def validate_taxonomy_second_pass(value: str) -> str:
    """Validate the taxonomy second-pass method."""
    if not isinstance(value, str) or value not in {"off", "llm", "agentic"}:
        raise ValueError("taxonomy_second_pass must be off, llm, or agentic")
    return value


def validate_taxonomy_agentic_max_turns(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("taxonomy_agentic_max_turns must be a positive integer")
    return value


@dataclass
class Config:
    harness: str = default_field("harness")
    analysis_outputs: list[str] = default_field("analysis_outputs")
    preflight_detectors: list[str] = default_field("preflight_detectors")
    analysis_focus: str = default_field("analysis_focus")
    skills_dir: str = default_field("skills_dir")
    skill_paths: list[str] = default_field("skill_paths")
    disabled_skill_paths: list[str] = default_field("disabled_skill_paths")
    analysis_harness: str = default_field("analysis_harness")
    top_x: int = default_field("top_x")
    max_rounds: int = default_field("max_rounds")
    # Extend ordinary discovery beyond the round target until this many validated
    # failure modes remain. 0 disables the minimum; explicit budgets still apply.
    min_failure_modes: int = default_field("min_failure_modes")
    # Opt-in discovery horizon; empty keeps the established round/quality policy.
    campaign_policy: dict[str, Any] = default_field("campaign_policy")
    usage_pricing: dict[str, Any] = default_field("usage_pricing")
    require_llm_detectors: bool = default_field("require_llm_detectors")
    discovery_wait_for_plateau: bool = default_field("discovery_wait_for_plateau")
    discovery_terminate_if_stalled: bool = default_field("discovery_terminate_if_stalled")
    discovery_stall_rounds: int = default_field("discovery_stall_rounds")
    sample_fail_default: int = default_field("sample_fail_default")
    sample_pass_default: int = default_field("sample_pass_default")
    max_sample_per_call: int = default_field("max_sample_per_call")
    prefer_informative: bool = default_field("prefer_informative")
    text_truncate_chars: int = default_field("text_truncate_chars")
    # Safety caps — evaluating unbounded parquet dumps has OOM/segfault'd.
    default_eval_limit: int = default_field("default_eval_limit")
    max_eval_limit: int = default_field("max_eval_limit")
    # 0 = auto-tune from free RAM at launch; set >0 to pin a size.
    eval_chunk_size: int = default_field("eval_chunk_size")
    # --- LLM detector scoring ---
    # One HTTP call per item makes a full-corpus LLM eval impractical, so LLM
    # detectors are scored on a stratified subsample and reported as estimates
    # with confidence intervals. 0 disables subsampling (score every row).
    # Mid-loop / capped eval only; mode=full runs a census. Saving does not evaluate.
    llm_eval_sample: int = default_field("llm_eval_sample")
    llm_eval_min_per_stratum: int = default_field("llm_eval_min_per_stratum")
    # Bounded concurrency for those calls; 1 = serial.
    llm_eval_workers: int = default_field("llm_eval_workers")
    llm_eval_max_workers: int = default_field("llm_eval_max_workers")
    llm_eval_queue_mib: int = default_field("llm_eval_queue_mib")
    # Independent behavior-judge examples per batch; 1 = serial, maximum 8.
    judge_workers: int = default_field("judge_workers")
    # Shared validation contract for discovery, review, and supported coverage.
    behavior_strictness: str = default_field("behavior_strictness")
    # The judge follows the discovery harness with its own model and effort.
    behavior_judge_model: str = default_field("behavior_judge_model")
    behavior_judge_harness: str = default_field("behavior_judge_harness")
    behavior_judge_effort: str = default_field("behavior_judge_effort")
    # Claude-only; Codex uses effort.
    behavior_judge_thinking: str = default_field("behavior_judge_thinking")
    behavior_judge_thinking_budget_tokens: int = default_field(
        "behavior_judge_thinking_budget_tokens")
    # `autofa preflight cheating` reviews run on the judge endpoint (judge_base_url, else the hub).
    cheating_check_model: str = default_field("cheating_check_model")
    library_root: str = default_field("library_root")
    model: str = default_field("model")
    thinking: str = default_field("thinking")
    effort: str = default_field("effort")
    thinking_budget_tokens: int = default_field("thinking_budget_tokens")
    # Path to local Claude Code binary (empty = auto-detect `claude` on PATH)
    claude_cli: str = default_field("claude_cli")
    codex_cli: str = default_field("codex_cli")
    # Legacy shared endpoint defaults, retained for existing configurations.
    hub_base_url: str = default_field("hub_base_url")
    hub_api_key_env: str = default_field("hub_api_key_env")
    feature_model: str = default_field("feature_model")
    judge_model: str = default_field("judge_model")
    taxonomy_first_pass: str = default_field("taxonomy_first_pass")
    taxonomy_second_pass: str = default_field("taxonomy_second_pass")
    taxonomy_agentic_max_turns: int = default_field("taxonomy_agentic_max_turns")
    # Auto prefers Decisions for fixed labels and Responses for generated text.
    feature_api: str = default_field("feature_api")
    judge_api: str = default_field("judge_api")
    # Any OpenAI-compatible API endpoint. Empty feature settings
    # inherit hub_*; empty judge settings inherit the feature endpoint.
    # api_key_env = "none" explicitly disables authentication (e.g. local servers).
    feature_base_url: str = default_field("feature_base_url")
    feature_api_key_env: str = default_field("feature_api_key_env")
    judge_base_url: str = default_field("judge_base_url")
    judge_api_key_env: str = default_field("judge_api_key_env")
    # Ordered opt-in fallback classifiers used by bounded discovery diagnostics.
    classifier_profiles: list[dict[str, Any]] = default_field("classifier_profiles")
    classifier_comparison_limit: int = default_field("classifier_comparison_limit")
    feature_request_options: dict[str, Any] = default_field("feature_request_options")
    judge_request_options: dict[str, Any] = default_field("judge_request_options")
    improvement_request_options: dict[str, Any] = default_field("improvement_request_options")
    # Optional EvalRDL-compatible API (run catalog + /export/autofa).
    data_api_base_url: str = default_field("data_api_base_url")
    data_api_name: str = default_field("data_api_name")
    # Compact exports omit tool observations. Load the bulk trace export by default.
    remote_full_traces: bool = default_field("remote_full_traces")
    # Shared AutoFA library service. Credentials come from env or the OS keychain.
    # Set an HTTP loopback origin to publish to the bundled local service instead.
    remote_library_base_url: str = default_field("remote_library_base_url")
    remote_library_token_env: str = default_field("remote_library_token_env")
    # cloud sends lightweight reports; server also transfers local trace data.
    remote_library_mode: str = default_field("remote_library_mode")
    # Local browser starts independently of the shared service. Empty uses the
    # loopback publication URL if selected, otherwise http://127.0.0.1:8765.
    local_library_base_url: str = default_field("local_library_base_url")

    # --- Detector field policy (anti-cheat / portability) ---
    # content  = only input / CoT / trace / answer (DEFAULT — no grader metadata)
    # metadata = allow judge_text / scoring JSON / etc. (labels still banned)
    # custom   = use allowed_fields list below
    detector_fields: str = default_field("detector_fields")
    allow_metadata: bool = default_field("allow_metadata")
    allowed_fields: list[str] = default_field("allowed_fields")
    extra_allowed_fields: list[str] = default_field("extra_allowed_fields")
    banned_fields: list[str] = default_field("banned_fields")

    # --- Optimization objective (Pareto axes; higher is better) ---
    # Default: maximize ppv@π₀ × task_recall × behavior_purity frontier.
    objective_axes: list[str] = default_field("objective_axes")
    # geometric_mean is the only supported aggregation.
    objective_rank: str = default_field("objective_rank")
    objective_weights: list[float] = default_field("objective_weights")
    # USD per million tokens keyed by model id. Empty uses the simulated rates.
    efficiency_model_prices: dict = default_field("efficiency_model_prices")

    # --- Creative exploration (String Seed of Thought — arXiv:2510.21150) ---
    # When true, discovery prompts supply a seeded string for the agent to
    # manipulate to diversify hypotheses; sample_items uses exploration_seed.
    exploration_ssot: bool = default_field("exploration_ssot")
    # Main seed for a deterministic round sequence; None / TOML "auto" derives
    # the main seed from the initial run ID. Continue retains the sequence.
    exploration_seed: int | None = default_field("exploration_seed")
    # When true, inject soft mechanism-template priors (seed→lens). Default false =
    # open-ended discovery (SSoT entropy + evidence only; no named failure catalog).
    exploration_templates: bool = default_field("exploration_templates")

    # Completed investigations without additional coverage before plateau stopping.
    discovery_plateau_patience: int = default_field("discovery_plateau_patience")

    harness_defaults: dict[str, dict[str, Any]] = default_field("harness_defaults")

    def __post_init__(self) -> None:
        from autofa.session.campaign_policy import validate_campaign_policy

        self.campaign_policy = validate_campaign_policy(self.campaign_policy)
        if self.usage_pricing:
            from autofa.session.usage import validate_pricing

            self.usage_pricing = validate_pricing(self.usage_pricing)
        validate_min_failure_modes(self.min_failure_modes)
        validate_discovery_wait_for_plateau(self.discovery_wait_for_plateau)
        validate_discovery_terminate_if_stalled(self.discovery_terminate_if_stalled)
        validate_discovery_stall_rounds(self.discovery_stall_rounds)
        validate_behavior_strictness(self.behavior_strictness)
        validate_objective_rank(self.objective_rank)
        from autofa.session.classifier_profiles import validate_profile_config

        validate_profile_config(self)
        from autofa.util.capacity import validate_config_capacity

        validate_config_capacity(self)
        if self.remote_library_mode not in {"cloud", "server"}:
            raise ValueError("remote_library_mode must be 'cloud' or 'server'")

    def resolved_library_root(self) -> Path:
        if self.library_root:
            return Path(self.library_root).expanduser().resolve()
        return library_root()


# Friendly aliases users may type in config.toml
_OBJECTIVE_ALIASES: dict[str, str] = {
    "precision": "ppv_at_pi0",
    "ppv": "ppv_at_pi0",
    "ppv_at_pi0": "ppv_at_pi0",
    "purity": "behavior_purity",
    "behavior_purity": "behavior_purity",
    "specificity": "behavior_specificity",
    "behavior_specificity": "behavior_specificity",
    "recall": "recall",
    "task_recall": "task_recall",
    "pct_caught": "pct_caught",
    "raw_ppv": "raw_ppv",
}


def normalize_objective_name(name: str | None) -> str:
    """Map user-facing aliases to canonical objective axis names."""
    raw = (name or "").strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")
    return _OBJECTIVE_ALIASES.get(key, raw)


def resolve_objective_axes(cfg: Config | None = None) -> list[str]:
    """Active weighted axes."""
    from autofa.objectives import weights_by_axis

    c = cfg or default_config()
    return [axis for axis, weight in weights_by_axis(c).items() if weight > 0]


def default_config() -> Config:
    return Config()


def project_root() -> Path | None:
    """Prefer an explicit project config, then this checkout, then the caller's repo."""
    here = Path.cwd().resolve()
    ancestors = [here, *here.parents]
    for root in ancestors:
        if (root / "autofa.toml").is_file():
            return root
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "src/autofa").is_dir():
        return checkout
    for root in ancestors:
        if (root / ".git").exists() or (root / "pyproject.toml").is_file():
            return root
    return None


def config_path(*, create_home: bool = True) -> Path:
    """Locate configuration, optionally preserving a nonexistent profile directory."""
    explicit = os.environ.get("AUTOFA_CONFIG")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if "AUTOFA_HOME" in os.environ:
        home = autofa_home() if create_home else global_env_path().parent
        return home / "config.toml"
    root = project_root()
    if root is not None:
        return root / "autofa.toml"
    home = autofa_home() if create_home else global_env_path().parent
    return home / "config.toml"


# Deployment defaults select services, LLM endpoints, and the initial harness.
# Personal analysis, filesystem paths, and credentials stay in user config.
_SITE_CONFIG_FIELDS = frozenset({
    "harness",
    "data_api_base_url", "data_api_name", "remote_full_traces",
    "remote_library_base_url", "remote_library_token_env", "remote_library_mode",
    "local_library_base_url", "hub_base_url", "hub_api_key_env",
    "feature_model", "judge_model", "feature_api", "judge_api",
    "feature_base_url", "feature_api_key_env",
    "judge_base_url", "judge_api_key_env", "feature_request_options", "judge_request_options",
    "improvement_request_options",
})


def site_config_path() -> Path | None:
    """Use an explicit deployment file or the preset beside this installed source.

    Never search the caller's working directory for deployment settings. An empty
    AUTOFA_SITE_CONFIG disables automatic preset loading, including in tests.
    """
    explicit = os.environ.get("AUTOFA_SITE_CONFIG")
    if explicit is not None:
        return Path(explicit).expanduser().resolve() if explicit.strip() else None
    checkout = Path(__file__).resolve().parents[2]
    preset = checkout / "config/internal.toml"
    if ((checkout / "pyproject.toml").is_file()
            and (checkout / "src/autofa").is_dir() and preset.is_file()):
        return preset
    return None


def _config_template() -> tuple[tomlkit.TOMLDocument, set[str]]:
    """Layer deployment values over public defaults; saved user values win later."""
    document = tomlkit.parse(defaults.defaults_text())
    path = site_config_path()
    if path is None:
        return document, set()
    try:
        with path.open("rb") as stream:
            overrides = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        # Do not echo file contents: a malformed preset could contain credentials.
        raise ValueError(f"Cannot read deployment defaults from {path}") from exc
    unknown = overrides.keys() - _SITE_CONFIG_FIELDS
    if unknown:
        raise ValueError(f"Unsupported deployment settings in {path}: {', '.join(sorted(unknown))}")
    if "harness" in overrides:
        from autofa.harness.settings import HARNESS_NAMES

        if not isinstance(overrides["harness"], str) or overrides["harness"] not in HARNESS_NAMES:
            raise ValueError(f"Deployment harness in {path} must be codex or claude")
    for key, value in overrides.items():
        document[key] = value
    return document, set(overrides)


def parse_exploration_seed(value: Any) -> int | None:
    """Normalize TOML exploration_seed: ``\"auto\"`` / empty / None → None (auto)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value == int(value):
        return int(value)
    s = str(value).strip().lower()
    if s in {"", "auto", "none", "null"}:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def config_source_path(path: Path | None = None) -> Path:
    """Locate the settings first launch would use without creating or migrating files."""
    explicit_path = path is not None
    selected = Path(path).expanduser() if explicit_path else config_path(create_home=False)
    legacy = Path.home() / ".autofa/config.toml"
    if (not selected.exists() and not explicit_path and not os.environ.get("AUTOFA_CONFIG")
            and "AUTOFA_HOME" not in os.environ and selected != legacy and legacy.is_file()):
        return legacy
    return selected


def ensure_default_config(*, require_harness: bool = True, path: Path | None = None) -> Path:
    """Create project config once, preserving legacy settings without deleting them."""
    source_path = config_source_path(path)
    explicit_path = path is not None
    path = Path(path).expanduser() if explicit_path else config_path()
    if not path.exists():
        if source_path != path and source_path.is_file():
            content = source_path.read_text(encoding="utf-8")
            # Reject an invalid legacy config before creating the project copy.
            tomllib.loads(content)
        else:
            from autofa.harness.settings import detect_default_harness

            document, site_fields = _config_template()
            harness = document["harness"]
            if require_harness and "harness" not in site_fields:
                harness = detect_default_harness(
                    codex_cli=document["codex_cli"] or None,
                    claude_cli=document["claude_cli"] or None,
                )
            document["harness"] = harness
            for key in ("model", "thinking", "effort", "thinking_budget_tokens"):
                document[key] = harness_default(harness, key, document)
            judge = document["behavior_judge_harness"]
            if judge == "auto":
                judge = harness
            for key in ("behavior_judge_model", "behavior_judge_thinking",
                        "behavior_judge_effort", "behavior_judge_thinking_budget_tokens"):
                document[key] = harness_default(judge, key, document)
            content = tomlkit.dumps(document)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)
        except FileExistsError:
            pass
    migrate_config_toml(path)
    return path


# Key variable that hub_base_url used before AutoFA stopped shipping a default endpoint.
LEGACY_HUB_API_KEY_ENV = "INFERENCE_HUB_API_KEY"  # pragma: allowlist secret


def migrate_config_toml(path: Path | None = None) -> bool:
    """Fill missing settings without inferring intent from explicitly saved values."""
    path = Path(path) if path is not None else config_path()
    if not path.is_file():
        return False
    raw = path.read_text(encoding="utf-8")
    try:
        document = tomlkit.parse(raw)
    except (ValueError, tomlkit.exceptions.ParseError):
        return False
    changed = _migrate_config_document(document)
    if changed:
        _write_config_atomically(path, tomlkit.dumps(document))
    return changed


def _migrate_config_document(document: tomlkit.TOMLDocument) -> bool:
    """Apply the shared legacy rules in memory for execution and inspection."""
    data = document.unwrap()
    template, site_fields = _config_template()
    # Missing harness in older files meant Claude; this is historical provenance,
    # not the default for new configurations (which comes from defaults.toml).
    saved_harness = data.get("harness", "claude")
    saved_judge_harness = data.get("behavior_judge_harness", template["behavior_judge_harness"])
    if saved_judge_harness == "auto":
        saved_judge_harness = saved_harness
    changed = False
    for key, value in template.items():
        if key in data:
            continue
        if key == "harness":
            value = saved_harness
        elif key in {"model", "thinking", "effort", "thinking_budget_tokens"}:
            value = harness_default(saved_harness, key, data)
        elif key in {"behavior_judge_model", "behavior_judge_thinking",
                     "behavior_judge_effort", "behavior_judge_thinking_budget_tokens"}:
            value = harness_default(saved_judge_harness, key, data)
        elif (key == "hub_api_key_env" and key not in site_fields
              and str(data.get("hub_base_url") or "").strip()):
            # A saved hub URL used this key name while it was the shipped default.
            # A deployment preset's own key name takes priority.
            value = LEGACY_HUB_API_KEY_ENV
        document[key] = value
        changed = True
    return changed


def _write_config_atomically(path: Path, content: str) -> None:
    """A failed migration must leave the previous configuration readable."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), path.stat().st_mode & 0o777 if path.exists() else 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_config(*, read_only: bool = False) -> Config:
    """Resolve settings, optionally without first-launch detection or file writes.

    Inspection uses the same normalization and validation as execution. Missing
    settings are layered exactly as config migration would, but inspecting a fresh
    installation does not authenticate a harness or create a profile.
    """
    path = config_source_path() if read_only else ensure_default_config()
    cfg = default_config()
    if read_only:
        if path.is_file():
            document = tomlkit.parse(path.read_text(encoding="utf-8"))
            _migrate_config_document(document)
        else:
            document, _ = _config_template()
        data = document.unwrap()
    else:
        with path.open("rb") as f:
            data = tomllib.load(f)
    for key, value in data.items():
        if not hasattr(cfg, key):
            continue
        if key == "exploration_seed":
            setattr(cfg, key, parse_exploration_seed(value))
        elif key == "objective_axes" and isinstance(value, list):
            setattr(cfg, key, [normalize_objective_name(str(a)) for a in value])
        else:
            setattr(cfg, key, value)
    from autofa.session.classifier_profiles import validate_profile_config

    validate_profile_config(cfg)
    cfg.analysis_outputs = validate_analysis_outputs(cfg.analysis_outputs, cfg.preflight_detectors)
    from autofa.session.campaign_policy import validate_campaign_policy

    cfg.campaign_policy = validate_campaign_policy(cfg.campaign_policy)
    if cfg.usage_pricing:
        from autofa.session.usage import validate_pricing

        cfg.usage_pricing = validate_pricing(cfg.usage_pricing)
    validate_min_failure_modes(cfg.min_failure_modes)
    validate_discovery_wait_for_plateau(cfg.discovery_wait_for_plateau)
    validate_discovery_terminate_if_stalled(cfg.discovery_terminate_if_stalled)
    validate_discovery_stall_rounds(cfg.discovery_stall_rounds)
    validate_behavior_strictness(cfg.behavior_strictness)
    from autofa.services import validate_remote_services

    validate_remote_services(cfg)
    if cfg.remote_library_mode not in {"cloud", "server"}:
        raise ValueError("remote_library_mode must be 'cloud' or 'server'")
    validate_taxonomy_first_pass(cfg.taxonomy_first_pass)
    validate_taxonomy_second_pass(cfg.taxonomy_second_pass)
    validate_taxonomy_agentic_max_turns(cfg.taxonomy_agentic_max_turns)
    from autofa.llm.endpoints import LLM_APIS

    if not isinstance(cfg.feature_api, str) or cfg.feature_api not in LLM_APIS:
        raise ValueError("feature_api must be auto, decisions, responses or chat_completions")
    if cfg.judge_api != "" and cfg.judge_api not in LLM_APIS:
        raise ValueError("judge_api must be blank or a supported LLM API")
    from autofa.util.capacity import validate_config_capacity

    validate_config_capacity(cfg)
    if not isinstance(cfg.behavior_judge_model, str) or not cfg.behavior_judge_model.strip():
        raise ValueError("behavior_judge_model must be a nonblank string")
    if not isinstance(cfg.cheating_check_model, str) or not cfg.cheating_check_model.strip():
        raise ValueError("cheating_check_model must be a nonblank string")
    from autofa.harness.settings import validate_harness_settings

    validate_harness_settings(cfg)
    validate_objective_rank(cfg.objective_rank)
    # Keep zero-weight axes and positional weight alignment in the saved config.
    resolve_objective_axes(cfg)
    # File references belong to this config, not whichever directory launches AutoFA.
    def local_path(raw: str) -> str:
        value = Path(raw).expanduser()
        return str((value if value.is_absolute() else path.parent / value).resolve())

    for key in ("skills_dir", "library_root"):
        if getattr(cfg, key):
            setattr(cfg, key, local_path(getattr(cfg, key)))
    cfg.skill_paths = [local_path(raw) for raw in cfg.skill_paths]
    return cfg


def _toml_value(value: Any) -> str:
    if isinstance(value, dict):
        return "{ " + ", ".join(
            f"{json.dumps(str(k))} = {_toml_value(v)}" for k, v in value.items()
        ) + " }"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(x) for x in value) + "]"
    return str(value)


@dataclass
class RunConfig:
    """Per-run dataset binding and discovery state."""

    run_id: str
    parquet_paths: list[str]
    model_id: str
    # Empty list = all benchmarks.
    benchmarks: list[str] = field(default_factory=list)
    harness: str = "claude"  # Historical identity when loading legacy run records.
    harness_settings: dict[str, Any] = field(default_factory=dict)
    # Resolved endpoint selected by a saved diagnostic comparison, never global config.
    classifier_profile_settings: dict[str, Any] = field(default_factory=dict)
    classifier_policy: dict[str, Any] = field(default_factory=dict)
    usage_pricing: dict[str, Any] = field(default_factory=dict)
    require_llm_detectors: bool = False
    campaign_policy: dict[str, Any] = field(default_factory=dict)
    campaign_state: dict[str, Any] = field(default_factory=dict)
    control_revision: int = 0
    control_last_request: str = ""
    top_x: int = default_field("top_x")
    max_rounds: int = default_field("max_rounds")
    # Saved runs without this field keep the historical minimum of one detector on Resume.
    min_failure_modes: int = 1
    # Saved runs without this field use the default Off policy on Resume.
    discovery_wait_for_plateau: bool = False
    # Existing runs without this field adopt the default On policy when resumed.
    discovery_terminate_if_stalled: bool = True
    # Legacy snapshots adopt three recovery rounds.
    discovery_stall_rounds: int = 3
    round: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    seed_library_path: str | None = None
    # Versioned cloud provenance. branch_key is the stable append identity;
    # source_run_id on each publication is always this run's current run_id.
    remote_library_id: str | None = None
    remote_branch_id: str | None = None
    remote_branch_key: str | None = None
    remote_version_id: str | None = None
    remote_base_version_id: str | None = None
    eval_limit: int | None = None
    # Original user-selected logs (gym folder / jsonl / parquet); display only.
    source_paths: list[str] = field(default_factory=list)
    source_kind: str = "parquet"
    # Harness that produced the analyzed logs (distinct from AutoFA's ``harness``).
    source_harness: str = "unknown"
    # Selected analyses: "efficiency", "agent", or both. Empty for preflight-only runs.
    analysis_outputs: list[str] = field(default_factory=list)
    # Snapshot for Resume. Older runs never opt into a newly enabled detector.
    preflight_detectors: list[str] = field(default_factory=list)
    # Data API provenance. Local parquet_path is retained only in the run config.
    reference_data: list[dict[str, str]] = field(default_factory=list)
    selection_id: str | None = None
    # Creative exploration (resolved at run start / advance_round).
    exploration_ssot: bool = default_field("exploration_ssot")
    exploration_seed: int = 0
    # Provenance only: whether Config.exploration_seed fixed the main seed at creation.
    exploration_seed_fixed: bool = False
    # Unset (None) resolves to the automatic main seed for run_id in __post_init__.
    exploration_base_seed: int | None = None
    exploration_sequence_index: int = 0
    exploration_templates: bool = default_field("exploration_templates")
    # Corpus domain detected at preflight; scopes the mechanism-template catalog.
    corpus_domain: str = "unknown"
    # Immutable guidance for Resume; source files are read only at run creation.
    analysis_focus: str = "model"  # Legacy saved runs predate harness-focused analysis.
    skill_snapshots: list[dict[str, str]] = field(default_factory=list)
    skills_root: str = ""
    skill_exclusions: list[str] = field(default_factory=list)
    skill_ignored_paths: list[str] = field(default_factory=list)
    # Harness reference packs left out because the analyzed harness did not match.
    skill_unmatched_packs: list[str] = field(default_factory=list)
    objective_settings: dict[str, Any] = field(default_factory=dict)
    # Legacy saved runs used mandatory causal validation. New runs explicitly
    # snapshot their selected policy in create(), including Continue launches.
    behavior_strictness: str = "causal"
    # Stable identity for all-model runs; older single-model runs may also have one.
    # model_id remains the actual row filter.
    dataset_id: str = ""

    def __post_init__(self) -> None:
        from autofa.session.campaign_policy import validate_campaign_policy

        if self.exploration_base_seed is None:
            from autofa.session.exploration import resolve_exploration_seed, round_exploration_seed

            self.exploration_base_seed = resolve_exploration_seed(run_id=self.run_id)
            self.exploration_seed = round_exploration_seed(
                self.exploration_base_seed, self.exploration_sequence_index,
            )
        self.campaign_policy = validate_campaign_policy(self.campaign_policy)
        validate_min_failure_modes(self.min_failure_modes)
        validate_discovery_wait_for_plateau(self.discovery_wait_for_plateau)
        validate_discovery_terminate_if_stalled(self.discovery_terminate_if_stalled)
        validate_discovery_stall_rounds(self.discovery_stall_rounds)
        validate_behavior_strictness(self.behavior_strictness)

    def library_model_id(self) -> str:
        """Name a selected model directly; use a stable ID for model mixtures."""
        # Multiple benchmarks/sources do not imply multiple models. Prefer the
        # selected model even if an older run persisted a mixture dataset_id.
        if self.model_id not in {"*", "ALL", ""}:
            return self.model_id
        if not self.dataset_id:
            from autofa.data.dataset_identity import dataset_mixture_id

            self.dataset_id = dataset_mixture_id(
                self.parquet_paths,
                model_id=self.model_id,
                benchmarks=self.resolved_benchmarks(),
                reference_data=self.reference_data,
            )
        if self.dataset_id:
            return self.dataset_id
        return "all-models"

    def resolved_benchmarks(self) -> list[str]:
        return list(self.benchmarks)

    def benchmark_label(self) -> str:
        bs = sorted(set(self.resolved_benchmarks()))
        if not bs:
            return "all"
        if len(bs) == 1:
            return bs[0]
        if len(bs) <= 3:
            return "+".join(bs)
        return f"{len(bs)}-benchmarks"

    def dataset_label(self) -> str:
        """Short dataset name from bound parquet path(s)."""
        model = self.library_model_id()
        if self.dataset_id:
            return model
        if not self.parquet_paths:
            return "—"
        p = Path(self.parquet_paths[0])
        # Prefer parent folder when it's a model/dump dir; else file stem.
        if p.is_file() or p.suffix == ".parquet":
            parent = p.parent.name
            stem = p.stem
            if parent and parent not in {".", ""}:
                return parent if parent != stem else stem
            return stem
        return p.name

    def explorer_title(self) -> str:
        """Header for /dashboard: run id · model · dataset/benchmark."""
        model = self.library_model_id()
        bench = self.benchmark_label()
        ds = self.dataset_label()
        # Avoid repeating the model name twice when folder == model_id.
        if ds in {model, "—", ""}:
            scope = f"{model} / {bench}"
        else:
            scope = f"{model} / {bench} · {ds}"
        return f"run {self.run_id} · {scope}"

    def run_dir(self) -> Path:
        d = runs_root() / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def publication_branch_key(self) -> str:
        """Stable branch identity for publishing this run."""
        if self.remote_branch_key:
            return str(self.remote_branch_key)
        if self.remote_library_id:
            raise ValueError(
                "Remote provenance has a library id but no branch key; "
                "select the cloud version again"
            )
        return self.run_id

    def save(self) -> Path:
        from autofa.session.campaign_policy import validate_campaign_policy

        self.campaign_policy = validate_campaign_policy(self.campaign_policy)
        validate_min_failure_modes(self.min_failure_modes)
        validate_discovery_wait_for_plateau(self.discovery_wait_for_plateau)
        validate_discovery_terminate_if_stalled(self.discovery_terminate_if_stalled)
        validate_discovery_stall_rounds(self.discovery_stall_rounds)
        validate_behavior_strictness(self.behavior_strictness)
        self.library_model_id()
        path = self.run_dir() / "config.json"
        from autofa.util.atomic import atomic_write_json

        atomic_write_json(path, asdict(self))
        return path

    @classmethod
    def load(cls, run_id: str) -> RunConfig:
        path = runs_root() / run_id / "config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("remote_base_version_id") and data.get("remote_version_id"):
            data["remote_base_version_id"] = data["remote_version_id"]
        data["reference_data"] = _backfill_reference_data(data)
        if str(data.get("source_harness") or "").strip() in {"", "unknown"}:
            data["source_harness"] = summarize_source_harness(
                reference.get("source_harness")
                for reference in data["reference_data"]
            )
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def create(
        cls,
        parquet_paths: list[Path],
        model_id: str,
        benchmarks: list[str] | None,
        cfg: Config,
        seed_library_path: str | None = None,
        *,
        seed_source_run_id: str | None = None,
        remote_library_id: str | None = None,
        remote_branch_id: str | None = None,
        remote_branch_key: str | None = None,
        remote_version_id: str | None = None,
        remote_base_version_id: str | None = None,
        max_rounds: int | None = None,
        min_failure_modes: int | None = None,
        top_x: int | None = None,
        eval_limit: int | None = None,
        source_paths: list[str] | list[Path] | None = None,
        source_kind: str = "parquet",
        source_harness: str = "unknown",
        reference_data: list[dict[str, str]] | None = None,
        selection_id: str | None = None,
    ) -> RunConfig:
        from autofa.session.guidance import snapshot_guidance

        references = normalize_reference_data(reference_data)
        # Each source of a mixed run can select its own harness reference pack.
        guidance = snapshot_guidance(cfg, source_harnesses=[
            source_harness, *(reference["source_harness"] for reference in references),
        ])
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        benches = list(benchmarks or [])
        # Default evaluate subsample — never leave unlimited on multi-GB dumps.
        el = eval_limit if eval_limit is not None else int(setting_value(cfg, "default_eval_limit"))
        el = max(1, min(el, int(setting_value(cfg, "max_eval_limit"))))
        src = [_normalized_source_path(p) for p in (source_paths or [])]
        from autofa.session.exploration import resolve_exploration_seed, round_exploration_seed

        ssot = bool(setting_value(cfg, "exploration_ssot"))
        templates = bool(setting_value(cfg, "exploration_templates"))
        override = setting_value(cfg, "exploration_seed")
        fixed = override is not None
        base_seed = resolve_exploration_seed(
            run_id=run_id, round_idx=0, override=int(override) if fixed else None
        )
        sequence_index = 0
        if seed_library_path:
            state = _continued_exploration_state(Path(seed_library_path))
            if state.get("base_seed") is not None:
                base_seed = state["base_seed"]
            sequence_index = state["sequence_index"] + 1
        seed = round_exploration_seed(base_seed, sequence_index)
        branch_key = remote_branch_key or run_id
        base_version_id = remote_base_version_id or remote_version_id
        from autofa.objectives import snapshot_objectives

        harness_settings = snapshot_harness_settings(cfg)
        run = cls(
            run_id=run_id,
            parquet_paths=[str(p.resolve()) for p in parquet_paths],
            model_id=model_id,
            benchmarks=benches,
            harness=cfg.harness,
            analysis_outputs=list(cfg.analysis_outputs),
            preflight_detectors=list(cfg.preflight_detectors),
            harness_settings=harness_settings,
            campaign_policy=json.loads(json.dumps(cfg.campaign_policy)),
            usage_pricing=json.loads(json.dumps(cfg.usage_pricing)),
            require_llm_detectors=cfg.require_llm_detectors,
            top_x=int(top_x) if top_x is not None else cfg.top_x,
            max_rounds=int(max_rounds) if max_rounds is not None else cfg.max_rounds,
            min_failure_modes=(
                cfg.min_failure_modes if min_failure_modes is None else min_failure_modes
            ),
            discovery_wait_for_plateau=cfg.discovery_wait_for_plateau,
            discovery_terminate_if_stalled=cfg.discovery_terminate_if_stalled,
            discovery_stall_rounds=cfg.discovery_stall_rounds,
            seed_library_path=seed_library_path,
            remote_library_id=remote_library_id,
            remote_branch_id=remote_branch_id,
            remote_branch_key=branch_key,
            remote_version_id=remote_version_id,
            remote_base_version_id=base_version_id,
            eval_limit=el,
            source_paths=src,
            source_kind=str(source_kind or "parquet"),
            source_harness=str(source_harness or "unknown").strip() or "unknown",
            reference_data=references,
            selection_id=selection_id,
            exploration_ssot=ssot,
            exploration_seed=seed,
            exploration_seed_fixed=fixed,
            exploration_base_seed=base_seed,
            exploration_sequence_index=sequence_index,
            exploration_templates=templates,
            objective_settings=snapshot_objectives(cfg),
            behavior_strictness=cfg.behavior_strictness,
            **guidance,
        )
        from autofa.session.budget_transfer import inherit_continuation_budget
        from autofa.session.classifier_profiles import initialize_classifier_policy
        from autofa.session.recovery import run_lock
        from autofa.session.usage import initialize_run_accounting

        directory = runs_root() / run.run_id
        directory.mkdir(parents=True, exist_ok=False)
        try:
            with run_lock(directory), inherit_continuation_budget(
                run, seed_path=seed_library_path, source_run_id=seed_source_run_id,
            ):
                run.save()
                if "agent" in run.analysis_outputs:
                    initialize_classifier_policy(cfg, run, new_run=True)
                initialize_run_accounting(directory, pricing=run.usage_pricing or None,
                                          before_first_dispatch=True)
        except BaseException:
            # No paid work is dispatched by creation. Remove this unique partial
            # run so a later launch never mistakes failed setup for historical work.
            # A transferred source is kept intact if the final journal write fails.
            if not getattr(run, "_budget_handoff_committed", False):
                shutil.rmtree(directory, ignore_errors=True)
            raise
        return run


def _continued_exploration_state(seed_path: Path) -> dict[str, int | None]:
    """Prefer the portable cursor; libraries without one continue past their rounds."""
    from autofa.core.library import FunctionLibrary

    library = FunctionLibrary.load(seed_path.expanduser())
    if library.exploration_state is not None:
        return library.exploration_state
    # Libraries without a cursor (such as efficiency-only publications) use any
    # recorded discovery round, advancing at least once instead of restarting.
    rounds = [fn.meta.get("round") for fn in library.functions]
    return {"base_seed": None,
            "sequence_index": max([0, *(value for value in rounds if type(value) is int)])}


def snapshot_harness_settings(cfg: Config) -> dict[str, Any]:
    from autofa.harness.settings import behavior_judge_settings, harness_model

    judge = behavior_judge_settings(cfg)
    return {
        "harness": cfg.harness,
        "model": harness_model(cfg.harness, cfg.model, cfg=cfg),
        "thinking": cfg.thinking,
        "effort": cfg.effort,
        "thinking_budget_tokens": cfg.thinking_budget_tokens,
        "behavior_judge_harness": judge["harness"],
        "behavior_judge_model": judge["model"],
        "behavior_judge_thinking": judge["thinking"],
        "behavior_judge_effort": judge["effort"],
        "behavior_judge_thinking_budget_tokens": judge["thinking_budget_tokens"],
    }


def restore_harness_settings(cfg: Config, run: RunConfig) -> Config:
    fields = {
        "harness",
        "model",
        "thinking",
        "effort",
        "thinking_budget_tokens",
        "behavior_judge_harness",
        "behavior_judge_model",
        "behavior_judge_thinking",
        "behavior_judge_effort",
        "behavior_judge_thinking_budget_tokens",
    }
    for name, value in run.harness_settings.items():
        if name in fields:
            setattr(cfg, name, value)
    return cfg


def _normalized_source_path(value: str | Path) -> str:
    """Preserve remote references while normalizing local source paths."""
    text = str(value).strip()
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        parsed = urllib.parse.SplitResult("", "", "", "", "")
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return text
    return str(Path(text).expanduser().resolve())


def _backfill_reference_data(data: dict[str, Any]) -> list[dict[str, str]]:
    """Recover Data API provenance from parquet ``source.json`` sidecars when none is recorded."""
    existing = normalize_reference_data(data.get("reference_data"))
    if existing:
        return existing
    parquets = [str(path) for path in (data.get("parquet_paths") or []) if path]
    recovered: list[dict[str, str]] = []
    for parquet in parquets:
        sidecar = Path(parquet).expanduser().parent / "source.json"
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        run_id = str(payload.get("run_id") or "").strip()
        base_url = str(payload.get("base_url") or "").strip().rstrip("/")
        export_path = str(payload.get("export_path") or "").strip()
        catalog = payload.get("catalog")
        catalog = catalog if isinstance(catalog, dict) else {}
        if not run_id or not base_url or not export_path.startswith("/"):
            continue
        recovered.append(
            make_reference_data(
                run_id=run_id,
                export_url=f"{base_url}{export_path}",
                source_harness=str(catalog.get("harness") or "unknown"),
                parquet_path=parquet,
                run_uuid=catalog.get("run_uuid"),
            )
        )
    return normalize_reference_data(recovered)


def list_runs() -> list[str]:
    root = runs_root()
    return sorted(
        (p.name for p in root.iterdir() if p.is_dir() and (p / "config.json").exists()),
        reverse=True,
    )

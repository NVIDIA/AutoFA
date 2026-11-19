# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find trusted local dependencies for artifact inspection and prepare speech models.

AutoFA installs no system or Python packages: LibreOffice, PyAV and the speech
packages are the user's own installation. Document bytes never select executables;
only an explicit speech setup request downloads a model.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from pathlib import Path

from autofa.util.cancellation import check_cancelled

# Some Linux distributions' LibreOffice packages also install audio and video codecs,
# so the user installs LibreOffice with the documented commands.
OFFICE_SETUP = (
    "https://github.com/NVIDIA/AutoFA/blob/main/docs/artifact-dependencies.md"
    "#office-pages-and-slides"
)
# Audio, video and speech need a PyAV that the user builds against their own FFmpeg.
AUDIO_VIDEO_SETUP = (
    "https://github.com/NVIDIA/AutoFA/blob/main/docs/artifact-dependencies.md"
    "#audio-video-and-speech"
)
_OFFICE_ENV = "AUTOFA_LIBREOFFICE"


def office_executable() -> str | None:
    """Resolve LibreOffice on PATH, a standard app install, or an explicit path."""
    override = os.environ.get(_OFFICE_ENV)
    if override:
        candidate = Path(override).expanduser()
        return (
            str(candidate)
            if candidate.is_absolute() and candidate.is_file() and os.access(candidate, os.X_OK)
            else None
        )
    for name in ("libreoffice", "soffice"):
        if executable := shutil.which(name):
            return executable
    system = platform.system()
    candidates: list[Path] = []
    if system == "Darwin":
        candidates = [
            Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
            Path.home() / "Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]
    elif system == "Windows":
        candidates = [
            Path(root) / "LibreOffice/program/soffice.exe"
            for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)")
            if (root := os.environ.get(variable))
        ]
    elif system == "Linux":
        candidates = [Path("/usr/bin/soffice"), Path("/opt/libreoffice/program/soffice")]
    return next(
        (str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)),
        None,
    )


def office_support_status() -> dict:
    """Report LibreOffice, which the user installs; AutoFA never installs it."""
    executable = office_executable()
    result: dict = {
        "capability": "office",
        "status": "available" if executable else "unavailable",
        "executable": executable,
        "limitations": [
            "Office visuals are print-layout conversions. Read native text/cells as well "
            "for hidden or non-printing content; rendered formulas may be recalculated."
        ],
    }
    if not executable:
        result.update(
            reason=(
                "AUTOFA_LIBREOFFICE does not name an executable file."
                if os.environ.get(_OFFICE_ENV)
                else "LibreOffice was not found. AutoFA does not install it; install it "
                "as described in the setup guide, or set AUTOFA_LIBREOFFICE to the "
                "absolute soffice path of an existing installation."
            ),
            setup=OFFICE_SETUP,
        )
    return result


def _media_support(capability: str) -> dict:
    """Report the PyAV (and faster-whisper) that the user installs; AutoFA never installs them.

    PyAV's PyPI wheels bundle an FFmpeg with codecs AutoFA must not distribute, and
    faster-whisper requires PyAV, so both come from the user's own source build.
    """
    packages = {"av": "PyAV"} if capability == "audio" else {
        "av": "PyAV", "faster_whisper": "faster-whisper",
    }
    missing = [name for module, name in packages.items()
               if importlib.util.find_spec(module) is None]
    result: dict = {"capability": capability,
                    "status": "unavailable" if missing else "available"}
    if missing:
        result.update(
            missing=missing,
            reason=(
                f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not installed. "
                "AutoFA does not install PyAV, FFmpeg or speech packages; build them as "
                "described in the setup guide."
            ),
            setup=AUDIO_VIDEO_SETUP,
        )
    return result


def prepare_artifact_support(
    capability: str,
    *,
    install: bool = False,
    model: str = "base",
) -> dict:
    """Inspect dependencies, or explicitly download a fixed speech model.

    Office and audio only report LibreOffice and the user-built PyAV. Speech can
    download a local model once the user has installed PyAV and faster-whisper.
    """
    check_cancelled()
    if capability not in {"office", "audio", "speech"}:
        raise ValueError("capability must be 'office', 'audio', or 'speech'.")
    if type(install) is not bool:
        raise ValueError("install must be a boolean.")
    if capability == "office":
        return office_support_status()
    if capability == "speech" and model not in {"tiny", "base", "small"}:
        raise ValueError("model must be 'tiny', 'base', or 'small'.")
    result = _media_support(capability)
    if capability == "speech":
        from autofa.data.artifact_audio import prepare_speech_model, speech_model_identity

        result["dependency_status"] = result["status"]
        try:
            model_state = (
                prepare_speech_model(model)
                if install and result["status"] == "available"
                else speech_model_identity(model)
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            # Without the packages, repeating install=True cannot help; point to setup instead.
            reason = str(exc) if result["status"] == "available" else (
                "Install PyAV and faster-whisper first (see setup); this tool cannot install "
                "them. It can download the model once they are installed."
            )
            model_state = {"status": "unavailable", "name": model, "reason": reason}
        result["model"] = model_state
        if model_state.get("status") != "available":
            result["status"] = "unavailable"
        result["limitations"] = [
            "Speech transcripts are machine-generated evidence and can be wrong. "
            "Music, sound effects and speaker identity are not assessed."
        ]
    # Only a speech model can be prepared here; packages and LibreOffice are the user's.
    if result["status"] != "available" and "setup" not in result:
        result["next_action"] = {
            "tool": "prepare_artifact_support",
            "arguments": {
                "capability": capability,
                "install": True,
                **({"model": model} if capability == "speech" else {}),
            },
        }
    return result

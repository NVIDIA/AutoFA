# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find and prepare trusted local dependencies for artifact inspection.

Document bytes never select executables or installation commands. Installing a
system dependency is explicit; ordinary reads and renders only inspect availability.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from autofa.util.cancellation import check_cancelled

OFFICE_INSTALL_TIMEOUT = 600
# Audio, video and speech need a PyAV that the user builds against their own FFmpeg.
AUDIO_VIDEO_SETUP = (
    "https://github.com/NVIDIA/AutoFA/blob/main/docs/artifact-dependencies.md"
    "#audio-video-and-speech"
)
_OFFICE_INSTALL_LOCK = threading.Lock()
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


def _office_install_command() -> list[str] | None:
    system = platform.system()
    if system == "Darwin" and (brew := shutil.which("brew")):
        return [brew, "install", "--cask", "libreoffice"]
    if system == "Windows" and (winget := shutil.which("winget")):
        return [
            winget,
            "install",
            "--id",
            "TheDocumentFoundation.LibreOffice",
            "--exact",
            "--silent",
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--disable-interactivity",
        ]
    if system == "Linux":
        prefix: list[str] = []
        if os.geteuid() != 0:
            sudo = shutil.which("sudo")
            if not sudo:
                return None
            # Never prompt for a password inside an agent tool.
            prefix = [sudo, "-n"]
        if apt := shutil.which("apt-get"):
            return [*prefix, apt, "install", "-y", "libreoffice", "fonts-dejavu-core"]
        if dnf := shutil.which("dnf"):
            return [*prefix, dnf, "install", "-y", "libreoffice"]
    return None


def office_support_status() -> dict:
    executable = office_executable()
    return {
        "capability": "office",
        "status": "available" if executable else "unavailable",
        "executable": executable,
        "install_command": _office_install_command() if not executable else None,
        "setup": (
            "LibreOffice renders DOCX, XLSX and PPTX pages. Use "
            "prepare_artifact_support(capability='office', install=True) to install it, "
            "or install LibreOffice with your OS package manager. "
            "For an existing custom installation, set AUTOFA_LIBREOFFICE to the "
            "absolute soffice executable path."
        ),
        "limitations": [
            "Office visuals are print-layout conversions. Read native text/cells as well "
            "for hidden or non-printing content; rendered formulas may be recalculated."
        ],
    }


def _stop_installer(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def _run_installer(command: list[str], *, timeout: float) -> dict:
    """Bound package-manager work and keep cancellation responsive."""
    check_cancelled()
    with tempfile.TemporaryFile() as output:
        env = dict(os.environ, HOMEBREW_NO_AUTO_UPDATE="1", DEBIAN_FRONTEND="noninteractive")
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=os.name == "posix",
        )
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                check_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _stop_installer(process)
                    return {"status": "unavailable", "reason": "Dependency installation timed out."}
                try:
                    process.wait(timeout=min(0.25, remaining))
                except subprocess.TimeoutExpired:
                    pass
            check_cancelled()
            output.seek(0, os.SEEK_END)
            output.seek(max(0, output.tell() - 4000))
            return {
                "status": "installed" if process.returncode == 0 else "unavailable",
                "returncode": process.returncode,
                "diagnostic": output.read().decode("utf-8", errors="replace").strip(),
            }
        finally:
            _stop_installer(process)


def install_office_support() -> dict:
    """Install the fixed LibreOffice package when explicitly requested by a tool."""
    check_cancelled()
    while not _OFFICE_INSTALL_LOCK.acquire(timeout=0.25):
        check_cancelled()
    try:
        check_cancelled()
        status = office_support_status()
        if status["status"] == "available":
            return dict(status, installed=False)
        if os.environ.get(_OFFICE_ENV):
            return dict(status, reason="AUTOFA_LIBREOFFICE does not name an executable file.")
        command = status["install_command"]
        if not command:
            return dict(status, reason="No supported noninteractive package manager is available.")
        try:
            installation = _run_installer(command, timeout=OFFICE_INSTALL_TIMEOUT)
        except OSError as exc:
            installation = {"status": "unavailable", "reason": str(exc)}
        result = office_support_status()
        result.update(installed=result["status"] == "available", installation=installation)
        if result["status"] != "available":
            result["reason"] = (
                "LibreOffice is still unavailable. Inspect installation diagnostics; "
                "the host may require an administrator to install it."
            )
        return result
    finally:
        _OFFICE_INSTALL_LOCK.release()


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
    """Inspect dependencies, or explicitly prepare a fixed supported capability.

    Office installs LibreOffice. Audio only reports the user-built PyAV; speech can
    download a local model once the user has installed PyAV and faster-whisper.
    """
    check_cancelled()
    if capability not in {"office", "audio", "speech"}:
        raise ValueError("capability must be 'office', 'audio', or 'speech'.")
    if type(install) is not bool:
        raise ValueError("install must be a boolean.")
    if capability == "office":
        return install_office_support() if install else office_support_status()
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
    # Only LibreOffice and a speech model can be prepared here; packages are the user's.
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

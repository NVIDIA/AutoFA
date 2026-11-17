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
import sys
import tempfile
import threading
import time
from pathlib import Path

from autofa.util.cancellation import check_cancelled

OFFICE_INSTALL_TIMEOUT = 600
_OFFICE_INSTALL_LOCK = threading.Lock()
_PYTHON_INSTALL_LOCK = threading.Lock()
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


def _python_install_command(package: str) -> list[str] | None:
    # Never alter a host Python or use pip's global/user installation fallback.
    if sys.prefix == sys.base_prefix:
        return None
    if uv := shutil.which("uv"):
        return [uv, "pip", "install", "--python", sys.executable, package]
    if importlib.util.find_spec("pip"):
        return [sys.executable, "-m", "pip", "install", package]
    return None


def _python_support(capability: str, *, install: bool) -> dict:
    module, package = {
        "audio": ("av", "av>=14"),
        "speech": ("faster_whisper", "faster-whisper>=1.1,<2"),
    }[capability]
    available = importlib.util.find_spec(module) is not None
    command = _python_install_command(package) if not available else None
    result = {
        "capability": capability,
        "status": "available" if available else "unavailable",
        "install_command": command,
    }
    if available or not install:
        return result
    if not command:
        return dict(
            result, reason="Activate AutoFA's virtual environment with uv or pip available."
        )
    while not _PYTHON_INSTALL_LOCK.acquire(timeout=0.25):
        check_cancelled()
    try:
        check_cancelled()
        # Another parallel artifact request may have just installed it.
        importlib.invalidate_caches()
        if importlib.util.find_spec(module):
            return dict(result, status="available", installed=False, install_command=None)
        try:
            installation = _run_installer(command, timeout=OFFICE_INSTALL_TIMEOUT)
        except OSError as exc:
            installation = {"status": "unavailable", "reason": str(exc)}
        importlib.invalidate_caches()
        available = importlib.util.find_spec(module) is not None
        return dict(
            result,
            status="available" if available else "unavailable",
            installed=available,
            install_command=None if available else command,
            installation=installation,
        )
    finally:
        _PYTHON_INSTALL_LOCK.release()


def prepare_artifact_support(
    capability: str,
    *,
    install: bool = False,
    model: str = "base",
) -> dict:
    """Inspect dependencies, or explicitly prepare a fixed supported capability."""
    check_cancelled()
    if capability not in {"office", "audio", "speech"}:
        raise ValueError("capability must be 'office', 'audio', or 'speech'.")
    if type(install) is not bool:
        raise ValueError("install must be a boolean.")
    if capability == "office":
        return install_office_support() if install else office_support_status()
    if capability == "speech" and model not in {"tiny", "base", "small"}:
        raise ValueError("model must be 'tiny', 'base', or 'small'.")
    result = _python_support(capability, install=install)
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
            model_state = {"status": "unavailable", "name": model, "reason": str(exc)}
        result["model"] = model_state
        if model_state.get("status") != "available":
            result["status"] = "unavailable"
        result["limitations"] = [
            "Speech transcripts are machine-generated evidence and can be wrong. "
            "Music, sound effects and speaker identity are not assessed."
        ]
    if result["status"] != "available":
        result["next_action"] = {
            "tool": "prepare_artifact_support",
            "arguments": {
                "capability": capability,
                "install": True,
                **({"model": model} if capability == "speech" else {}),
            },
        }
    return result

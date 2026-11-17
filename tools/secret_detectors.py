# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoFA's detect-secrets plugins for what the built-in detectors miss.

`.secrets.baseline` loads this file by its repository-relative path, so detect-secrets and
`tools/check_secrets.py` must run from the repository root.
"""

from __future__ import annotations

import re
from collections.abc import Generator

from detect_secrets.plugins.base import RegexBasedDetector

# Hosts under nvidia.com that are public and may appear in tracked files. Listing public hosts
# instead of internal ones keeps internal names out of the repository. Add a host only after
# confirming it is reachable without NVIDIA network access or credentials.
PUBLIC_NVIDIA_HOSTS = frozenset({
    "build.nvidia.com",
    "catalog.ngc.nvidia.com",
    "developer.nvidia.com",
    "docs.nvidia.com",
    "integrate.api.nvidia.com",
    "ngc.nvidia.com",
    "www.nvidia.com",
})


class NvidiaApiKeyDetector(RegexBasedDetector):
    """NVIDIA API keys, which start with `nvapi-`."""

    secret_type = "NVIDIA API Key"  # pragma: allowlist secret
    denylist = (re.compile(r"(?<![A-Za-z0-9_-])(nvapi-[A-Za-z0-9_-]{20,})"),)


class NvidiaHostDetector(RegexBasedDetector):
    """Hostnames under nvidia.com other than the public hosts listed above."""

    secret_type = "Non-public NVIDIA Hostname"  # pragma: allowlist secret
    # A subdomain is required, so addresses such as name@nvidia.com do not match.
    denylist = (re.compile(r"(?<![A-Za-z0-9.-])((?:[A-Za-z0-9-]+\.)+nvidia\.com)(?![A-Za-z0-9-])",
                           re.IGNORECASE),)

    def analyze_string(self, string: str) -> Generator[str, None, None]:
        """Yield each nvidia.com hostname in `string` that is not a listed public host."""
        for host in super().analyze_string(string):
            if host.lower() not in PUBLIC_NVIDIA_HOSTS:
                yield host

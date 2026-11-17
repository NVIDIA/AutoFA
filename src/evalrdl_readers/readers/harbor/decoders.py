# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from evalrdl_readers.core import Decoder

# ----------------------------------------------------------------------- decoders
# The reward key was renamed between adapter versions; chosen per result.json.

REWARD_V46 = Decoder(
    "rewards-composite",
    because="`verifier_result.rewards` carries a `composite_score` rollup alongside the per-criterion keys",
)
REWARD_V47 = Decoder(
    "rewards-reward",
    because="`verifier_result.rewards` carries a single `reward` key and nothing else",
)

CTRF = Decoder("ctrf", because="`verifier/ctrf.json` is a CTRF test report")
SCALEAI_OUT = Decoder(
    "runscript-output",
    because="`verifier/output.json` is the swebench-pro run-script report",
)
RCA_REPORT = Decoder(
    "criteria-report",
    because="`verifier/report.json` is the scaled-evals criteria report",
)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Imported last by the reader forkserver: how many threads it has once everything is loaded.

Every child it forks inherits this number, and refuses to read when it
is above one.
"""

from .procs import thread_count

THREADS = thread_count()

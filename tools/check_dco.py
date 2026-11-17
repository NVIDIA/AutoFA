# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate author sign-offs from paginated GitHub pull-request commits."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

SIGNOFF = re.compile(r"Signed-off-by:\s*(\S.*?)\s*<([^<>\s]+@[^<>\s]+)>\s*$", re.I)


def has_author_signoff(message: str, author_email: str) -> bool:
    trailers = subprocess.check_output(
        ["git", "interpret-trailers", "--parse"], input=message, text=True
    )
    return bool(author_email) and any(
        match and match.group(2).casefold() == author_email.casefold()
        for line in trailers.splitlines()
        if (match := SIGNOFF.fullmatch(line))
    )


def is_bot_commit(item: dict, bot_login: str | None) -> bool:
    """Whether GitHub attributes the commit to the bot account `bot_login`."""
    account = item.get("author") or {}
    return bool(bot_login) and account.get("login") == bot_login and account.get("type") == "Bot"


def validate_pages(
    pages: list[list[dict]], expected_count: int, exempt_bot: str | None = None
) -> list[str]:
    """Return one error per commit without its author's sign-off.

    Commits GitHub attributes to `exempt_bot` are skipped. The workflow passes it only for
    pull requests that the bot opened from a branch in this repository.
    """
    commits = [commit for page in pages for commit in page]
    if expected_count < 1 or len(commits) != expected_count:
        return ["Incomplete commit list; split the PR or obtain a complete commit review."]
    errors = []
    for item in commits:
        if is_bot_commit(item, exempt_bot):
            continue
        commit = item["commit"]
        if not has_author_signoff(commit["message"], commit["author"]["email"]):
            errors.append(
                f"{item['sha'][:12]}: missing the commit author's DCO Signed-off-by trailer"
            )
    return errors


def main() -> int:
    """Read commit pages from stdin; return 0 if every commit is signed off or exempt, else 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--exempt-bot", help="bot login whose own commits need no sign-off")
    args = parser.parse_args()
    pages = json.load(sys.stdin)
    errors = validate_pages(pages, args.expected_count, args.exempt_bot)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print("Read CONTRIBUTING.md before adding your own sign-off.", file=sys.stderr)
        return 1
    exempt = sum(is_bot_commit(item, args.exempt_bot) for page in pages for item in page)
    signed = args.expected_count - exempt
    note = f"; {exempt} by {args.exempt_bot} need none" if exempt else ""
    print(f"Verified author sign-offs on {signed} external contribution commits{note}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small native supervision surface; help and status never initialize an analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autofa.util.tool_result import result_failed as _result_failed


def main(argv: list[str] | None = None) -> int:
    from autofa.session.campaign_policy import CampaignBoundaryReached
    from autofa.session.supervision import (
        SupervisorInterrupted,
        checkpoint_run,
        finalize_run,
        lifecycle_request,
        resolve_run_dir,
        run_status,
        submit_steering,
    )

    parser = argparse.ArgumentParser(prog="autofa supervise",
                                     description="Inspect and supervise saved AutoFA runs")
    actions = parser.add_subparsers(dest="action", required=True)
    tools_parser = actions.add_parser("tools", help="Inspect registered native tool schemas")
    tools_parser.add_argument("--name", help="Inspect one registered tool")
    invoke = actions.add_parser("invoke", help="Execute a native tool on an inactive run; "
                                "may perform inference or mutate saved results")
    invoke.add_argument("--run", required=True)
    invoke.add_argument("--tool", required=True)
    invoke.add_argument("--arguments-file", type=Path, required=True)
    status = actions.add_parser("status", help="Read saved status; no runtime, inference or writes")
    status.add_argument("--run", required=True)
    status.add_argument("--full-snapshots", action="store_true",
                        help="Include full saved snapshot records (may be very large)")
    cost = actions.add_parser("cost", help="Read saved tokens, costs and round projections")
    cost_source = cost.add_mutually_exclusive_group(required=True)
    cost_source.add_argument("--run", help="Saved AutoFA run ID")
    cost_source.add_argument("--usage-dir", type=Path,
                             help="Standalone workflow's ledger directory")
    pricing = actions.add_parser("pricing", help="Set explicit USD rates, or a separate "
                                 "live-safe token reference with --estimate-only")
    pricing_source = pricing.add_mutually_exclusive_group(required=True)
    pricing_source.add_argument(
        "--run", help="Saved run ID; must be inactive unless --estimate-only")
    pricing_source.add_argument("--usage-dir", type=Path,
                                help="Existing standalone workflow ledger directory")
    pricing.add_argument("--settings-file", type=Path, required=True)
    pricing.add_argument("--estimate-only", action="store_true",
                         help="Set reference prices without changing ledger costs or spend "
                              "admission; permitted while the analysis is active")
    reconcile = actions.add_parser("reconcile-cost", help="Resolve a retained in-flight "
                                   "reservation from explicit provider accounting")
    reconcile.add_argument("--run", required=True)
    reconcile.add_argument("--id", required=True, help="Retained reservation ID")
    reconcile.add_argument("--cost-usd", type=float, required=True)
    reconcile.add_argument("--source", required=True, help="Accounting evidence/source reference")
    steer = actions.add_parser("steer", help="Queue discovery guidance at tool/turn boundaries")
    steer.add_argument("--run", required=True)
    steer.add_argument("--message-file", type=Path, required=True)
    steer.add_argument("--id", help="Stable receipt ID for retry reconciliation")
    finalize = actions.add_parser("finalize", help="Validate saved detectors without new discovery")
    finalize.add_argument("--run", required=True)
    checkpoint = actions.add_parser("checkpoint", help="Checkpoint an inactive run under its lock")
    checkpoint.add_argument("--run", required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    configure = actions.add_parser("configure", help="Set scoped worker/queue settings on an "
                                   "inactive run; preserve global settings and model identity")
    configure.add_argument("--run", required=True)
    configure.add_argument("--settings-file", type=Path, required=True)
    update = actions.add_parser("update", help="Queue live discovery goals/limits for the "
                                "owning run; preserves the active process and saved evidence")
    update.add_argument("--run", required=True)
    update.add_argument("--settings-file", type=Path, required=True)
    update.add_argument("--id", help="Stable request ID; retries do not add rounds again")
    update.add_argument("--expected-revision", type=int)
    contribute = actions.add_parser("contribute", help="Queue an attributed native candidate, "
                                    "repair or comparison at a safe boundary of a live run")
    contribute.add_argument("--run", required=True)
    contribute.add_argument("--tool", required=True)
    contribute.add_argument("--arguments-file", type=Path, required=True)
    contribute.add_argument("--id")
    contribute.add_argument("--expected-revision", type=int)
    contribute.add_argument("--library-sha256", help="Reject if the saved library changed")
    contribute.add_argument("--after-update", help="Depend on this accepted update receipt")
    request = actions.add_parser("request", help="Persist cooperative lifecycle intent")
    request.add_argument("--run", required=True)
    request.add_argument("intent", choices=("pause", "stop", "finalize", "release"),
                         help="Release explicitly permits work again; does not start a process")
    for command in (update, contribute, steer, request):
        command.add_argument("--fence-file", type=Path,
                             help="Campaign owner generation and goal revision binding")
    args = parser.parse_args(argv)
    try:
        if args.action == "tools":
            from autofa.mcp.server import TOOL_CALLABLES, _tool_input_schema

            if args.name and args.name not in TOOL_CALLABLES:
                raise ValueError(f"Unknown tool: {args.name}")
            result = {name: {"description": fn.__doc__, "input_schema": _tool_input_schema(fn)}
                      for name, fn in TOOL_CALLABLES.items() if not args.name or name == args.name}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        directory = (args.usage_dir if args.action in {"cost", "pricing"} and
                     args.usage_dir is not None
                     else resolve_run_dir(args.run))
        fence = None
        if getattr(args, "fence_file", None) is not None:
            from autofa.session.control import verify_fence

            fence = json.loads(args.fence_file.read_text())
            verify_fence(fence, protective=args.action == "request" and
                         args.intent in {"pause", "stop"})
        if args.action == "status":
            result = run_status(directory, full_snapshots=args.full_snapshots)
        elif args.action == "cost":
            from autofa.session.usage import usage_summary

            result = usage_summary(directory)
        elif args.action == "pricing":
            from autofa.session.usage import set_usage_pricing

            result = set_usage_pricing(directory, json.loads(args.settings_file.read_text()),
                                       estimate_only=args.estimate_only)
        elif args.action == "reconcile-cost":
            from autofa.session.admission import reconcile_reservation

            result = reconcile_reservation(directory, args.id, args.cost_usd, args.source)
        elif args.action == "steer":
            result = submit_steering(directory, args.message_file.read_text(), receipt_id=args.id,
                                     fence=fence)
        elif args.action == "update":
            from autofa.session.control import submit_update

            result = submit_update(directory, json.loads(args.settings_file.read_text()),
                                   request_id=args.id, expected_revision=args.expected_revision,
                                   fence=fence)
        elif args.action == "contribute":
            from autofa.session.control import submit_contribution

            result = submit_contribution(directory, args.tool,
                                         json.loads(args.arguments_file.read_text()),
                                         request_id=args.id,
                                         expected_revision=args.expected_revision,
                                         library_sha256=args.library_sha256, fence=fence,
                                         after_update=args.after_update)
        elif args.action == "checkpoint":
            result = checkpoint_run(directory, args.output)
        elif args.action == "finalize":
            result = finalize_run(args.run)
        elif args.action == "configure":
            from autofa.session.run_capacity import set_run_capacity

            result = set_run_capacity(args.run, json.loads(args.settings_file.read_text()))
        elif args.action == "invoke":
            import asyncio
            import inspect

            from autofa.config import RunConfig, load_config, restore_harness_settings
            from autofa.mcp.runtime import load_runtime_from_run
            from autofa.mcp.server import TOOL_CALLABLES, invoke_tool
            from autofa.session.recovery import run_lock
            from autofa.session.supervision import check_lifecycle

            arguments = json.loads(args.arguments_file.read_text())
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            if args.tool not in TOOL_CALLABLES:
                raise ValueError(f"Unknown native tool: {args.tool}")
            try:
                inspect.signature(TOOL_CALLABLES[args.tool]).bind(**arguments)
            except TypeError as exc:
                raise ValueError(f"Invalid arguments for {args.tool}: {exc}") from exc
            check_lifecycle(directory)
            with run_lock(directory):
                run = RunConfig.load(args.run)
                load_runtime_from_run(run, restore_harness_settings(load_config(), run))
                result = asyncio.run(invoke_tool(args.tool, arguments, caller="driver"))
        else:
            result = lifecycle_request(directory, args.intent, fence=fence)
    except CampaignBoundaryReached as exc:
        result = exc.as_result()
    except SupervisorInterrupted as exc:
        result = exc.as_result()
    except (OSError, ValueError, RuntimeError) as exc:
        from autofa.session.control import FenceBusy
        from autofa.session.supervision import SupervisionBusy
        from autofa.util.model_errors import safe_error_detail

        result = {"ok": False, "error": safe_error_detail(str(exc))}
        if isinstance(exc, (FenceBusy, SupervisionBusy)):
            result.update(status="busy", retryable=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if _result_failed(result) else 0

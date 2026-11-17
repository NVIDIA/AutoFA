---
name: library
description: List detectors in the working library. Use for /library.
---

# /library

Handled **locally** (no agent turn). Prefer the `/library` slash command.

If you must list from a tool call: `list_library_functions` (includes `failure_modes_summary`).
Before proposing new modes, always call `list_failure_modes` as well — it includes dropped
modes that are no longer in the working library.

Use `list_helpers` / `get_helper` for the separate unscored helper catalog. Helpers are
ordinary reusable functions; their count never contributes to the detector target.

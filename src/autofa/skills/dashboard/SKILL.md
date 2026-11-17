---
name: dashboard
description: Open the interactive failure explorer. Use for /dashboard.
---

# /dashboard

Prefer the **local** `/dashboard` command (handled outside the agent) which opens
an interactive arrow-key explorer:

1. ↑/↓ select a detector
2. →/Enter expand → see **buckets** with PRECISION (fires → % fail) bars
3. Expand a bucket → example fail ids
4. ← collapse · Esc quit

If you must summarize without the TUI, call `list_library_functions` and print
per-detector `Flagged | Coverage | Normalized precision | Detector adherence` (respectively
fraction flagged, recall, `ppv@π₀` and `behavior_purity`), plus each detector's
`buckets` (n, n_fail, precision%). Precision is failure association; Detector adherence is
claim support under the selected Judge Policy, not the proportion of flagged rollouts that fail.

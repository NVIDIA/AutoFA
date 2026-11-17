---
name: goal
description: Check the current discovery target and stopping decision. Use for /goal.
---

# /goal

Handled **locally** (no agent turn). Prefer the `/goal` slash command.

If you must check from a tool call:
1. `pareto_status` — read the active `target_grade`, the qualifying count and remaining
   requirements, `goal_met`, and `ready_for_final_validation`. The raw number of detectors
   does not establish target completion; frontier trends are supplementary diagnostics.
2. If exploration is still required and progress stalls, pivot within the current round.
   Follow `autofa_protocol.md`'s round-completion rules before `advance_round`.
3. In regular discovery, return the summary when `goal_met` is true. When
   `ready_for_final_validation` is true, follow `final_validation_guidance`: perform only
   an explicitly requested full scan, or return the summary for driver-managed final scoring.
   Explicit cancellation always applies.
4. Recommend `/save` + `/dashboard` when wrapping up

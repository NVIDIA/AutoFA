# Hermes — Architecture & loop

**At a glance:** A turn enters at `run_conversation`, builds a prologue (sanitize → system-prompt restore → preflight compression → plugin hooks), then runs an **outer iteration loop** bounded by an `IterationBudget` (**default 90 parent / 50 subagent**) wrapping an **inner per-call retry loop** (`api_max_retries`, default 3). `finish_reason` drives continuation; `finalize_turn` handles budget exhaustion with one extra toolless summary call.

## How it works

**Entry & prologue.**
- Entry: `run_conversation` in `agent/conversation_loop.py`.
- Prologue (once per turn): runs stdio guarding, surrogate sanitization, system-prompt restore/build, crash-persistence, **preflight compression**, the `pre_llm_call` plugin hook, and external-memory prefetch.

**Outer loop (iterations).**
Each iteration: interrupt check → increment call count → consume from `IterationBudget`. `execute_code` turns are **refunded** back to the budget. `_budget_grace_call` grants one extra call past the cap, then forces exit regardless of outcome.

**Inner loop (per-call retries).** Handles `finish_reason`, invalid responses, and format-recovery restarts. See [error-taxonomy.md](error-taxonomy.md) for the retry/backoff policy.

**`finish_reason` handling:** `"stop"` → done; `"length"`/`"incomplete"` → continuation retry with a nudge; `"content_filter"` → terminal content-policy block.

**Finalization.** On budget exhaustion, one extra **toolless** summary call is made; kanban workers get `_record_task_failure(outcome="timed_out")`.

**Alternate runtimes.** `api_mode == "codex_app_server"` hands the whole turn to a Codex subprocess; `provider == "moa"` uses the Mixture-of-Agents loop (`agent/moa_loop.py`).

**Key defaults.** **Parent `max_iterations` default 90**, **subagent `delegation.max_iterations` default 50**. Diagnostic `_turn_exit_reason` strings: `"interrupted_by_user"`, `"budget_exhausted"`, `"max_iterations_reached(N/M)"`.

## Where errors can originate

- **Iteration-cap truncation:** turn ends without a natural answer once the count reaches `max_iterations` (90) or budget hits 0; a fallback toolless summary call is made — which may itself fail.
- **Subagents can exceed the parent budget:** each subagent has an independent 50-cap, so parent+children total work can exceed 90.
- **Grace-call ambiguity:** the single `_budget_grace_call` runs even with budget exhausted, then forces exit regardless of outcome.
- **User interrupt mid-loop:** `agent._interrupt_requested` breaks the loop, leaving a partial turn.

## Sources

- `nousresearch/hermes-agent` `agent/conversation_loop.py` — entry, outer/inner loops, finish_reason handling.
- `nousresearch/hermes-agent` `agent/iteration_budget.py`, `agent/turn_finalizer.py` — iteration caps (90/50), grace call, finalization.

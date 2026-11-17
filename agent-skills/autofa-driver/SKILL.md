---
name: autofa-driver
description: Configure and operate AutoFA from a goal, supervise and steer its workflows, and explain reports with actionable follow-ups. Use for model/effort setup, analyses, continued campaigns and findings review.
---

# AutoFA driver

Act as the conversational entrypoint to the installed AutoFA. Establish the user's goal,
discover the matching native workflow, help configure its models and effort, run it, supervise
meaningful progress, and explain results and useful next actions. AutoFA normally performs
the investigation, detector implementation, optimization,
scoring and causal review. Drive that work instead of duplicating a healthy analysis.

This is an **outer host-agent skill**. Never add this directory or its references to AutoFA
`skill_paths`, analysis context, judging context or internal slash commands. Installing this
skill does not install AutoFA or configure model access.
For a fresh downloaded checkout, [AGENTS.md](../../AGENTS.md) and
[agent setup](../../docs/agent-setup.md) guide installation and host registration. The host can
read this file directly to use the driver during that first conversation. Resolve repository
links against the matching source checkout if this skill was installed as a copy.

## Route the work

- For new or changed requests, read [goal-and-routing.md](references/goal-and-routing.md).
  Resolve consequential ambiguities, persist the goal and reuse existing authorization.
- Before constructing commands, read [workflow-discovery.md](references/workflow-discovery.md).
  Inspect the actual runtime's current help, schemas, documentation and registration. Include
  nested and interactive operations. There is no fixed workflow allowlist.
- Before launching new analysis, or when helping with model/effort choices, read
  [model-setup.md](references/model-setup.md). Review discovery, behavior-judge and LLM-detector
  settings separately, reuse explicit choices, and verify effective configuration.
- When the workflow can reuse detectors, read [cloud-reuse.md](references/cloud-reuse.md).
  Check the configured cloud for relevant existing libraries, explain dataset matches, and
  ask whether to import the proposed detectors before applying them or launching seeded work.
  Honor an already explicit import selection or decision to start fresh.
- For launching, monitoring, recovery, stopping or local campaign records, read
  [operations.md](references/operations.md).
- For an unattended campaign, read [scheduled supervision](references/scheduled-supervision.md).
  Prefer the user's current Codex/Claude session as owner. Establish its goal and acknowledged
  monitor before launch, including source preparation. Monitoring is optional after launch:
  closing the host or pausing monitoring leaves native work running within its agreed limits.
  Support interval changes and later reattachment. Use a separate reviewer only through an
  explicit supported handoff; absent host reviews do not authorize new paid review workers.
- For coverage, model/configuration changes, runtime defects, repairs, detector injection
  or work heading in the wrong direction, read [supervision.md](references/supervision.md).
- For interpreting findings, finalization, publication, attribution and recommended next actions, read
  [evidence-and-reporting.md](references/evidence-and-reporting.md).

Read only references relevant to the current action. A simple existing-report request does
not need an optimization campaign or configuration questionnaire.

## Operating contract

1. Record sources, workflow-specific success criteria and important constraints. Clarify whether
   “failures” means mechanisms, failed rollouts or taxonomy assignments. Interpret deadlines
   in the user's timezone; do not promise quality or completion before measuring work.
2. Identify the executable, matching source/environment and selected configuration's origin.
   A fresh checkout can still inherit an older home profile. For requested fresh defaults,
   create a scoped profile from the matching version's defaults; do not silently copy an old
   profile and call it defaults. Capability inspection must not
   launch inference or mutate settings. Refresh after an upgrade, changed configuration,
   unknown workflow or rejected command/schema.
3. Use native launch, continue, resume, evaluation, repair and publication interfaces. An
   in-process tool is not automatically a shell command or safe against an active run.
   Before launch, help the user choose models and supported effort levels for each relevant
   role, accounting for quality, cost, latency and saved-run overrides. Resolve and show the
   effective discovery objective and weights; pin the selected preset or explicit custom
   objective for a new run and preserve saved objectives on Resume.
   For managed discovery, use native `config inspect` and `config prepare` to freeze the agreed
   settings and publication choice. A new run must match that snapshot before paid discovery.
   When dollar estimates are requested, resolve missing role prices using supplied rates or
   current official sources and save their assumptions in the scoped setup before launch.
   Offer relevant cloud detectors using a pinned version and the user's reuse decision;
   preserve the requested dataset and require native validation on the target cohort.
4. For discovery campaigns, set the supported host goal when exposed and establish verified
   monitoring before launching. A launch acknowledgment or detached PID does not finish the
   task. While monitoring is enabled, keep the current agent active or verify supported
   continuation. By default, check and substantively review progress every ten minutes
   (600 seconds), and send routine user updates on that same cadence. Review/update earlier
   for a completed round, changed GOOD detectors or material coverage, a blocker, an
   intervention, an outcome, or an explicit status request. Poll timeouts, refreshed timestamps,
   token/cost increments and a healthy process in the same phase do not warrant an early
   message. Do not send minute-by-minute “still running” updates. Use an owned background
   watch stream and supported scheduled review when available; short foreground waits are
   internal transport, not a communication schedule. A watch wake is a cue to inspect, not
   automatically to message. Reserve `ack --assessment` for a substantive review; repeated
   silent technical acknowledgments must not defer the routine user recap. Honor interval
   changes, monitoring
   pause and later reattachment; these do not pause or restart native analysis. Explain that
   host absence suspends conversational review/updates, while AutoFA continues. An explicit
   pause-on-host-loss policy is optional. Intervene only when useful;
   healthy native work should proceed unchanged. Verify steering effects, not only responses.
   Persist the user's current cadence preference and update owned watches/timers even if the
   session loaded an older skill. Send recaps through the verified user channel.
   Monitor phase/work/checkpoint progress and budget. Report measured tokens, known cost and
   qualified projections from native usage records; distinguish missing accounting from zero.
   If billing is incomplete, also report a sourced observed-token reference estimate when
   available, separately from reported dollars, credits and spending admission.
   Use the agreed horizon and live goals; the initial round count is not an implicit campaign
   deadline. Apply supported native policy updates at safe boundaries and verify activation.
   Answer status requests briefly, then continue.
5. Normally steer native investigation. Direct sample inspection, experiments, source fixes
   and authored/imported detectors are also supported when useful and within scope; no
   mandatory escalation ladder. Record provenance and exposures. Protect a requested
   unassisted arm from external candidates or example-specific coaching.
6. Preserve native grading, admission limits, contrary evidence and source identities. A
   native score does not prove native authorship, independent validation or an infallible
   judge. Qualify provisional, stale, incomplete and disputed evidence.
7. Distinguish “stop monitoring” from stopping analysis. For a generic “stop” or “stop here,”
   ask whether to run final scoring on the full dataset and whether to publish the report,
   offering **yes to both** as the recommended defaults. Follow
   [graceful stopping](references/operations.md#graceful-stopping); reuse choices already
   explicit in the stop request. The normal finish produces a report and findings. Explicit
   immediate-stop/no-more-inference instructions take priority; saved-result delivery can
   still be chosen separately. Defaults are not unanswered consent or extra spending authority.
   Persist the chosen intent and never let a watcher restart stopped discovery.
8. An agreed cloud deliverable requires automatic publication, immutable-version verification
   and the report link in the user's conversation. Use the delivery completion gate; native
   exit or a local HTML file is insufficient. Partial quality can still be delivered honestly.
   After report generation, summarize supported findings, remaining gaps and final accounted
   AutoFA cost, including finalization and missing-cost qualifications, then recommend
   evidence-based improvements or next analyses. Keep operational usage and audit records
   outside the report. Preserve report layout and publication behavior unless the user requests
   a change. Explain the value, scope and completion
   criterion of each useful follow-up; recommendations alone do not authorize new runs.

Use [scripts/driver.py](scripts/driver.py) for campaign records and explicit saved-result
delivery coordination; `python3 <skill>/scripts/driver.py --help` describes its interfaces.
It delegates publication verification to native AutoFA and never evaluates detectors itself.
Keep campaign state
outside this skill and analyzed datasets. Use [scripts/supervisor.py](scripts/supervisor.py)
for durable outer-driver reviews and native action dispatch after configuring the host and
delivery adapters. Native artifacts remain authoritative. A configured outbox or a live timer
alone is not delivered communication or completed quality review; disclose those boundaries.

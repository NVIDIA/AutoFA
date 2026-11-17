---
name: validate
description: Evaluate all library detectors on the dataset and refresh scores. Use for /validate.
---

# /validate

<!-- AUTOFA_BEHAVIOR_POLICY -->

Automatic round-end/final validation selects OK-or-better candidates from the latest current
checkpoint. Promising candidates missing adherence receive bounded review before selection.
Weak, No matches, and nonqualifying unresolved candidates retain their existing scores and
history with an explicit skip reason; a meaningful revision can make them eligible again.
Intentional exclusions are not pending final work. This selection does not restrict an
explicit request to evaluate named detectors or the whole library with `mode="full"`.
Keep sampled results labeled as estimates, including in the saved report.


When `refinement.mode='polishing'` follows 100% validation, the task is to improve existing
detectors to Good or better. Do not create detector names or start new family discovery.
Keep feedback compact: changed scores and evaluation scopes, remaining Good requirements,
a few evidence groups with counts and representative item IDs, then the general distinction
to test. Passing matches affect precision; reviewed counterexamples affect adherence; lost
failed matches affect coverage; missing or erroneous predictions need evidence recovery or
implementation repair. A passing example is not itself a rejected mechanism, and a benchmark
label is not a mechanism explanation. Cite available review reasons; do not invent them.
When cohorts differ, describe the revised estimate rather than claiming a measured regression.
Use ordinary repair/submission and capped evaluation. For detectors that can call an LLM,
the runtime checks the original fixed panel plus all cumulative problematic datapoints and
retained support controls, without a
half-panel cap. The union is targeted regression evidence, not population scoring. Only the
original panel supplies population estimates and provisional grades. Start with the summary and
retrieve relevant cases or pages as needed, preserving the claim's required evidence scope.
The runtime checks all retained controls; there is no need to load them all into the conversation.
Detectors with no LLM calls in their source, helpers or parents scan the full corpus by
default, deriving repair feedback from that same scan. Their population scores use the full
corpus; shared-panel search comparisons remain separate. An explicit debug limit still caps
the scan. Full prediction scoring does not require judging every example.
Use representative cases to test a general boundary, not task-ID or benchmark-specific exceptions.
Check it on task-diverse, previously uninspected positives and nearby contrasts when available
through existing tools. Preserve reserved independent-review evidence, and state when further
cases are unavailable; edited-on cases do not provide independent confirmation. This adds no
mandatory stage or review calls. Retain the controls across revisions. At phase close,
`advance_round(validation_response=...)` triggers 100% verification for selected revised
detectors and reuses unchanged results. The final report uses those validated scores.

1. `evaluate_functions` on the working library (optionally pass `names`).
   Inspect `judge_prefilter` first. A `skip` is an association-based scheduling decision:
   improve the detector and re-evaluate, or remove it with a reason. Do not call the judge
   again for unchanged skipped candidates. Missing/error/uncertain association evidence
   does not reject candidates. `skipped_pre_filter` means not judged, never zero purity.
2. Inspect `behavior_validation`. Default `submit_function` already evaluates and judges
   new/changed/stale versions with fires that survive the pre-filter.
   Call `judge_function` only for missing/stale
   reviews or a planned validation (runs a **fresh session in the selected judge harness**, not the
   NVIDIA hub). A missing purity value can mean UNKNOWN or processing error; retain those
   reviews and plan the appropriate follow-up instead of automatically repeating them.
   Full-corpus evaluation does not require a duplicate judgment of an unchanged current claim.
3. Inspect `behavior_validation` counts and `get_function_feedback` assessments.
   Contradicted conditions need detector/claim repair using cited counterexamples;
   unknown conditions need the specific missing evidence; processing errors need
   retry/repair. Low `behavior_purity` alone is not a mismatch. Keep unresolved
   modes open, and submit/re-judge any revised claim as a new version.
   Distinguish evidence for the actual claim from an inferred cause. Use the selected
   behavioral review standard; do not add mandatory final-cause proof to behavioral claims
   or waive evidence of continuing relevance in contextual claims. Explicit stronger causal
   assertions still require their full evidence in every mode. A broad family can encompass
   different subtypes when each reviewed example satisfies its shared definition.
   Specificity measures detail, not validity or a fourth standard objective. Follow the
   precision/recall preset for breadth without changing the selected review standard.
   Never improve prose by inventing a hidden belief. Avoid incidental conditions; frozen
   claim conditions remain mandatory. A revised claim needs fresh evaluation and review.
   Test repairs with `smoke_test_feature(item_ids=..., expected_outputs=...,
   expected_abstentions=...)`; author-selected controls do not replace independent validation.
   Call `get_function_repair` for grouped findings and next investigation calls. Before
   modifying a judged detector, record `plan_function_repair` with its finding/evidence IDs.
   Use `repair_implementation` for recorded detector execution errors, even when the reviewed
   claim is supported. Preserve its description, mechanism and relationships; smoke-test the
   failed cases and supported controls, then reevaluate and review the changed implementation.
   `retry_review` handles judge processing errors; transient provider faults need evaluation
   retries after restoring the service. Neither implies the claimed behavior was wrong.
   Validation then excludes recorded repair cases; processing retries preserve other verdicts.
   Use `defer_function_repair` for blocked evidence and `get_function_feedback` pagination
   for detailed assessments. Completing an attempt does not automatically promote a detector.
   If `claim_calibration` is `overclaim` / `underclaim` (even with decent purity):
   use `calibration_hint` as an editorial suggestion subject to the assessments above.
   A rewrite cannot silently waive acceptance conditions or validate unsupported causality.
4. `pareto_status` and summarize improvements vs the previous round.
5. Summarize with `Flagged | Coverage | Normalized precision | Detector adherence`,
   using recall for Coverage, `ppv@π₀` for Normalized precision and `behavior_purity`
   for Detector adherence. Include confirmed,
   rejected, uncertain, processing-error counts and decisive coverage, and mention
   per-bucket precision (fires → % fail). Precision measures failure association, while
   adherence measures whether the detector's claim is supported under the selected Judge
   Policy; these are different quantities. Tell the user to run `/dashboard`.

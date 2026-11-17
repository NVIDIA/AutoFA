---
name: save
description: Persist the working library to the run dir and durable catalog. Use for /save.
---

# /save

Handled **locally** (no agent turn). Prefer the `/save` slash command.

AutoFA saves the available results locally and publishes the current report. Saving
and publishing do not run detectors, start judge reviews, or retry failed reviews.
Sampled scores keep their labels, and completed inconclusive reviews keep unavailable
scores. During discovery, use capped evaluations and planned review repairs. Full-corpus
scoring runs at the runtime's validation checkpoints before discovery completion;
saving alone does not start it.

If you must save from a tool call: `library_save` with `to_catalog=true`, then report the saved paths.

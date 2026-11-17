---
name: help
description: Show AutoFA commands and metrics help. Use when the user says /help or asks what commands exist.
---

# /help

List the available AutoFA slash commands and explain the feedback metrics:
Grade, Flagged (share of all rollouts matched), Coverage (share of failed rollouts caught),
Normalized precision (`ppv@π₀`, π₀=0.30), Detector adherence (reviewed claim support),
and distinct failed-task support. Optimization uses fixed-panel distinct-task recall;
report Coverage uses attempt recall. Search weights do not change the Judge Policy.
Be concise. Prefer the built-in help table if shown; otherwise print the same command list yourself.

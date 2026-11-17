# Discover the installed workflows

For a checkout-based discovery request, compare the selected native source and this loaded
skill's source before setup. `config inspect` reports `provenance.source_checkout` and the
runtime identity. An installed skill can still point to an older worktree: read the matching
checkout's `AGENTS.md` and driver skill directly in that case. Use the documented setup/update
path for registration; a development request does not authorize silently reinstalling skills.

Build an evidence-backed map for the current request, not a handwritten catalog. Newly
registered workflows must be usable without editing this skill.

## Establish the runtime

Resolve executable/wrapper/shebang, interpreter/environment, installed module location,
source/release identity and configuration precedence. A nearby checkout and executable can
differ even with equal version labels. Read matching installer/wrapper metadata and docs.
Never print credentials or complete environment files. Follow that version's setup/readiness-check
path when needed before a costly analysis.

Establish configuration provenance as well as its selected path. A new checkout may import
a legacy home profile, or resolve a configuration from a parent directory or another editable
installation. Inspect this before first-launch creation, and follow
[model setup](model-setup.md#resolve-where-the-settings-came-from) for requested fresh defaults.

The helper fingerprints explicit runtime inputs without running or importing them:

```bash
python3 <skill>/scripts/driver.py identify --executable /absolute/path/to/autofa \
  --source-root /matching/release/or/checkout --config /path/to/config.json
```

It returns paths/hashes, not configuration contents. It **does not prove** that the source root
belongs to the executable; establish that binding through its wrapper/environment. For
module invocations, identify the interpreter and matching package tree, recording the complete
invocation in the map. Tree hashing belongs to discovery/refresh, not every health poll.

## Inspect without side effects

Inspect safe help/version/capability schemas and relevant nested help. Verify inspection
behavior first; if uncertain, read registration/implementation instead. Some programs mutate
state before processing help. Follow advertised command groups, dynamic registrations and
interactive/internal interfaces. Top-level help alone can be incomplete.

For each needed capability record:

- Purpose, entrypoint and surface: executable argument array, menu, slash command, MCP/API
  tool or supported adapter. Discovery in one surface does not make it callable in another.
- Accepted options/config schema, prerequisites, credentials, defaults and precedence.
- Outputs, identity, progress signals, completion/validation semantics and lifecycle actions.
- Side effects, including optional inference, and evidence paths to help/schema/docs/source.

An arbitrary private function is not a supported workflow. Prefer native interfaces; when a
necessary operation lacks one, identify the exact gap and use a small version-checked native
integration only within scope. Do not invent flags/signatures or load a live runtime mutably
for inspection. Map intent to native workflow semantics rather than forcing all work into
rounds or detector counts. Use native readiness checks/dry-run when appropriate. Discovery does not
authorize sending data, spending budget or publication.

## Cache and refresh

Store captured help/source evidence in campaign storage. `cache-put` accepts a map with a
nonempty `capabilities` list. Every entry has `name`, `surface`, `entrypoint`, `purpose`, and
an `evidence` list of paths; additional contracts are arbitrary JSON and names unrestricted.
The helper binds evidence hashes and runtime/source/config fingerprints; `cache-get` recomputes
them and fails on change. It never dispatches commands.

Refresh on source/runtime/config changes, new unknown behavior, missing evidence or command
mismatch. An unchanged cache proves only unchanged recorded evidence, not exhaustive discovery.
Inspect changed stop/save behavior even if command names stay the same. Keep secrets and raw
traces out of the map. Never use yesterday's map as an allowlist for future workflows.

# Set up AutoFA from the first agent message

This is the bootstrap for a host agent that the user has asked to set up or use the downloaded
AutoFA repository. It installs AutoFA and registers its conversational driver, then continues
with the user's task. It is not an AutoFA analysis skill or a requirement for code review.

## Start the driver immediately

Resolve the checkout containing [AGENTS.md](../AGENTS.md) and read
[agent-skills/autofa-driver/SKILL.md](../agent-skills/autofa-driver/SKILL.md) directly. Apply it
in the current conversation, reading its references only as needed. Registration below makes
the skill discoverable by name; it is not a prerequisite for following the same instructions
now. Do not tell the user to restart or send a second message just to begin driver setup.

If the user only supplied a path and requested setup, finish setup and ask for their goal.
If their message includes data and an analysis goal, carry that request into the driver after
setup instead of ending with “installed.” Installation alone does not authorize analysis,
paid probes or publication.

## Install the checkout

Use the checkout as the working directory for installation and AutoFA commands. Inspect
`pyproject.toml`, the existing environment and any user-specified interpreter/tool paths first.
Reuse working tools. If `uv` is missing, install it using an appropriate user-level method from
the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
Use the installed executable's actual path if the current shell has not refreshed `PATH`.

For this checkout, the standard installation is:

```sh
uv sync --python 3.12 --inexact
uv run --no-sync autofa --help
uv run --no-sync python -c "import autofa; print(autofa.__file__)"
```

`uv` can supply Python 3.12 and creates the project environment. `--inexact` preserves extra
packages already installed in it, such as development tools; it does not upgrade dependencies
just to refresh them. Respect an existing environment override and verify which environment
will be changed. Check that the import belongs to this checkout, not another installation.
Use `uv run --no-sync autofa ...` from here after synchronization. Recheck dependencies when
the checkout's requirements change rather than reinstalling on every status request.

The help/import checks need no model credentials or inference. Preserve `.env`, `autofa.toml`,
`AUTOFA_CONFIG`, `AUTOFA_HOME` and saved runs; installation must not reset the user's profile.
If downloads or execution are unavailable, state the concrete blocker and complete the
remaining local setup. Distinguish installed files from a verified runnable environment.

Before the first command that creates configuration, resolve its source. A fresh GitHub
checkout can still inherit an older `~/.autofa/config.toml`; installation and configuration
are separate. Show the effective file and any inheritance instead of calling copied settings
fresh defaults. If the user requests current defaults, follow the driver's
[configuration-source procedure](../agent-skills/autofa-driver/references/model-setup.md#resolve-where-the-settings-came-from):
use a new campaign profile with an explicit path and the native initializer, preserving old
files and carrying over only agreed overrides and credential references.

## Register the driver with the current host

Identify the **host agent** for this conversation. It need not be the Codex/Claude harness
chosen to execute AutoFA analyses. Install the driver for that host only, using its advertised
skill roots and configured overrides. Inspect an existing `autofa-driver` registration first.

| Host | Current user skill location, unless the host specifies another |
|---|---|
| Codex | `~/.agents/skills/autofa-driver`; some installations advertise `$CODEX_HOME/skills` or `~/.codex/skills` instead. Use one supported location, not duplicate registrations. |
| Claude Code | `~/.claude/skills/autofa-driver` |
| Another agent | Its supported skill installation mechanism; direct reading remains available when it has no skill catalog. |

Both [Codex](https://learn.chatgpt.com/docs/build-skills) and
[Claude Code](https://code.claude.com/docs/en/skills) support symlinked skill directories.
Where supported, create a link named `autofa-driver` to this checkout's absolute
`agent-skills/autofa-driver` path. Create only the necessary parent directory. A link already
resolving to this source is a successful no-op. If links are unavailable, copy the entire
package, including `references/`, `scripts/` and `agents/`, using the host's supported method.
Keep the source checkout available and record its path; repository-documentation links from
a copied skill must be resolved against that checkout.

Do not blindly overwrite a different, customized or broken existing installation. Inspect it,
reuse it if appropriate, or resolve a replacement using the user's preferences. Continue with
this checkout's skill directly while that registration choice is pending. Verify the registered
`SKILL.md`, its supporting files and the link target or copied content. Report its exact location
and any unresolved conflict instead of claiming a different package is installed.

Use the host's supported refresh/discovery mechanism when available. If its named skill catalog
has not refreshed, keep using the already-read skill in this conversation and state that named
invocation is pending. Codex normally detects changes automatically; Claude Code supports
`/reload-skills` for a newly created skill root. Do not pass host slash commands to a shell.
Once discovered, use `$autofa-driver` in Codex or `/autofa-driver` in Claude Code. Local file
registration does not install into a different machine or a cloud-only host.

The canonical package is `agent-skills/autofa-driver/`. Do not copy it into repository `skills/`,
`src/autofa/skills/`, AutoFA's `skill_paths`, native prompts or judge context. Repository
`CLAUDE.md` imports `AGENTS.md` so the bootstrap is shared with Claude hosts that use that
[project instruction convention](https://code.claude.com/docs/en/memory#agentsmd).

## Check access and continue

Follow [model setup](../agent-skills/autofa-driver/references/model-setup.md) for harness, judge
and detector LLM endpoint choices. Reuse configured/authenticated harnesses; status commands such as
`codex login status` and `claude auth status` do not establish model access or quota. If the
selected CLI is missing, install it through its official supported method. See
[configuration](configuration.md#credentials-and-harness-selection) for executable overrides
and the Claude SDK's bundled CLI option.

Sign-in and API credentials can require user participation. Give the specific login action
or credential configuration location, then continue independent setup while waiting. Never
ask the user to paste secret values into the conversation. Check presence/reference only;
packages cannot supply account access. Configure only services needed for the selected
workflow; cloud access is optional for local analysis. See [LLM endpoints](llm-endpoints.md).

Summarize the checkout/environment, registered skill path or pending discovery state, and
remaining access needs. Then continue with the supplied goal using the driver: choose the
workflow, clarify meaningful constraints, offer relevant cloud detectors, review model/effort
and cost settings, and run within the agreed scope. A setup success is not an analysis result.

## Bind unattended supervision when the goal needs it

For a continuing campaign, follow the driver's
[scheduled supervision](../agent-skills/autofa-driver/references/scheduled-supervision.md).
Agree the native discovery/judge/API stack and the outer review model separately. Persist the
goal, deadline/timezone, resource scope, escalation/assistance preferences, update cadence and
delivery destination. Help resolve concrete missing access or unsupported model/API settings;
do not silently switch models or assume a harness login grants detector LLM endpoint access.

For discovery, configure host-owned monitoring before launch, with the current conversation's
identity and a durable goal. Use an exposed goal tool when available; otherwise retain the goal
in the campaign and use supported foreground/event waits. Do not invent a host API. The matching
native `config inspect`/`config prepare` commands freeze the agreed settings and publication
choice. Start the guard, observe and acknowledge it from the host, then launch the returned
specification under monitoring. Native bindings carry the exact launch identity, including
failures before a run ID exists. See the scheduled-supervision reference for executable steps.

Review actual work every ten minutes by default and on significant events. The user may
change the cadence, pause monitoring or close the host; native AutoFA continues within its
agreed limits. While monitoring is enabled, keep the agent active or verify supported
continuation; renew expiring watches. Reattach later to the existing campaign and process.
The detached reviewer remains an explicit alternative with a verified handoff.
Do not declare cloud-required discovery complete until native publication is verified and the
findings/link reach the user. A configured outbox is durable storage, not conversation delivery.

Use a host's supported persistent task mechanism when available, or the bundled managed CLI
worker. Never invent an application API or silently add external messaging destinations.
Explain concrete availability limits before promising unattended delivery, including a local
machine going to sleep. Native deadlines/spend admission and the outer deadline watchdog are
separate from a responsive review worker. Preserve user stop intent across restarts and use
native queued updates for live round/target/horizon changes.

# Contributing to AutoFA

AutoFA discovers failure modes in evaluation traces, builds reusable detectors,
and checks their quality. Contributions should make that behavior easier to
understand, reproduce, or improve.

External contributions are welcome within a **limited scope**, described in
[Contribution scope](#contribution-scope). AutoFA is a research preview, and its
maintainers decide its direction. Review the [license terms](#license-of-contributions),
[DCO sign-off requirements](#developer-certificate-of-origin) and
[maintainer roles](#maintainers-and-code-ownership) before preparing a contribution.
Development takes place in [NVIDIA/AutoFA](https://github.com/NVIDIA/AutoFA).
Submit changes through pull requests to `main`, including maintainer changes.

## Contribution scope

Pull requests for these changes are welcome without prior discussion:

- Fixes for reproducible bugs, with a regression test.
- Corrections and clarifications to documentation, examples, and error messages.
- New or improved tests for existing behavior.

Any other change needs an issue and a maintainer's agreement before work starts.
This includes, for example:

- New features, CLI commands, configuration options, or trace formats.
- New dependencies, including optional extras.
- Changes to saved-run, library, report, or publication formats, or to the meaning
  of a metric.
- Changes to detector execution limits, or to the Library server's network,
  authentication, or request-validation checks.
- Changes to analysis prompts and skills, to the bundled evalrdl packages, or
  large refactors.

These changes are not accepted:

- Changes that add credentials, private datasets, personal information, or internal
  service details.
- Third-party code without a stated origin and a compatible license.
- Unrelated reformatting or dependency updates.

Maintainers may close a pull request outside this scope or ask for an issue first.

## Questions, bugs, and security

Use the repository's issue tracker for ordinary bugs, feature requests, and
documentation questions. The issue forms ask for a minimal reproduction and
environment details. Use synthetic traces and redact credentials, personal
information, private datasets, and internal service details from all examples.

Report vulnerabilities privately using [SECURITY.md](SECURITY.md). Community
participation follows [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md); it also lists the
private conduct-reporting contact. The project does not yet have a dedicated
public support channel.

## License of contributions

Code-only files are contributed under Apache-2.0. Documentation, prompts/skills
and assets follow the top-level **CC-BY-4.0 AND Apache-2.0** terms in
[LICENSE](LICENSE). Preserve existing third-party copyright, license and
attribution notices; the project license does not replace third-party terms.

## Developer Certificate of Origin

External contributors must certify each commit under the
[Developer's Certificate of Origin 1.1](#full-dco-text). Read the complete certificate before
signing off. Your sign-off certifies the origin of the contribution and your right
to submit it, including the necessary employer authorization when applicable.

Use your name and email address in your Git author identity, then sign off:

```sh
git commit -s
```

Git adds a trailer such as:

```text
Signed-off-by: Your Name <your.email@example.com>
```

If you omitted your own sign-off on your latest commit, and can make the DCO
certification, add it with:

```sh
git commit --amend --no-edit -s
```

Each commit in an external contribution needs the author's sign-off. Do not sign
on someone else's behalf. A DCO sign-off is a commit-message trailer, distinct
from a cryptographic Git signature.

The compliance workflow validates sign-offs for pull requests whose author is
not identified by GitHub as an organization member or owner. GitHub reports only
public organization membership to the workflow, so NVIDIA contributors with a
private membership either make it public or sign off their commits.
Dependabot's version-update pull requests need no sign-off for the commits GitHub
attributes to `dependabot[bot]`: its trailer names `support@github.com` rather than its
commit author, and only Dependabot can open such a pull request from a branch in this
repository. Any other commit added to one still needs its author's sign-off.
Maintainers also review contribution provenance, including third-party work submitted through an
NVIDIA contributor's pull request. NVIDIA contributors follow the applicable
internal IP review process.

### Full DCO text

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Source notices and third-party material

For NVIDIA-authored code-only files, use this exact short header with the
applicable copyright year and the file's comment syntax:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Documentation, skill files and `.github/CODEOWNERS` do not require a per-file
copyright/license header.

SPDX-only notices are permitted for NVIDIA-authored code only. Preserve the full
upstream copyright and license notices in third-party code; do not replace them
with the NVIDIA header. Describe any new dependency or
copied/adapted material, its origin and license in the pull request, and update
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the accompanying license texts.
Maintainers must complete the applicable IP/license review before accepting it.

## Maintainers and code ownership

| Role | GitHub account | Responsibilities |
| --- | --- | --- |
| Default code owner | [@teelinsan](https://github.com/teelinsan) | Review changes across the repository and coordinate additional reviewers. |
| Default code owner | [@ka00ri](https://github.com/ka00ri) | Review changes across the repository and coordinate additional reviewers. |
| Maintainer | [@fgalko-oss](https://github.com/fgalko-oss) | Review changes across the repository and coordinate additional reviewers. |
| Maintainer | [@martinagvilas](https://github.com/martinagvilas) | Review changes across the repository and coordinate additional reviewers. |
| Maintainer | [@aligatti](https://github.com/aligatti) | Review changes across the repository and coordinate additional reviewers. |

Repository-wide code ownership is defined in [`.github/CODEOWNERS`](.github/CODEOWNERS).
GitHub automatically requests the listed owners on pull requests. Each pull request
requires approval from at least one code owner with write access other than its
author. Other maintainers can review changes but do not satisfy the code-owner
requirement. Being listed as a code owner does not grant repository permissions or
authorize a public release.

Only NVIDIA employees whose GitHub accounts are linked to NVIDIA single sign-on
(SSO) approve and merge pull requests, as NVIDIA's source code management
standard requires. External contributors cannot approve or merge, including
their own pull requests.

Changes to ownership and repository permissions require a maintainer review.

## Development setup

From the checkout, let `uv` set up Python 3.12 and the development dependencies:

```sh
uv run --python 3.12 --extra dev python -m pytest -n auto
```

`-n auto` runs the suite on all CPU cores with pytest-xdist; tests must not depend on
execution order or on state left by another test. Omit it when debugging a single test.

Audio and video tests need PyAV built against the FFmpeg from `tools/build_ffmpeg.sh`, as
described in [audio, video and speech](docs/artifact-dependencies.md#audio-video-and-speech).
Without it they are skipped; CI builds both and runs them.

Install the [pre-commit hooks](#pre-commit-hooks) once per clone, so each commit runs
the fast checks that CI runs:

```sh
uv run --python 3.12 --extra dev pre-commit install
```

Follow the checkout's README for configuration and model authentication. Keep
credentials out of tracked files; the [secrets scan](#secrets-scan) checks every
commit. Unit tests should use local fixtures and test
doubles rather than paid model calls or credentials for private services.

Use a temporary `AUTOFA_HOME` when testing the CLI or Library so development runs
cannot modify your saved research. CI uses an isolated profile and a test harness
probe; it does not authenticate to a model provider or run a live analysis.

## Coding guidelines

- Keep each change focused on one problem. Describe the current behavior, the
  desired behavior, and any compatibility impact before a substantial redesign.
- Follow nearby code and the repository's Ruff configuration: Python 3.12,
  100-character lines, organized imports, and descriptive `snake_case` names for
  functions and variables. Add type hints to new public interfaces and explain
  non-obvious behavior with short docstrings or comments.
- Keep reusable analysis, data access, and persistence logic separate from CLI
  presentation. Report failures with enough context to act on them, preserve
  recoverable local state, and release files, locks, clients, and servers reliably.
- Preserve saved-run, library, report, and publication formats. Describe and test
  migrations or intentional compatibility changes. Do not silently rewrite user
  data or change the meaning of an existing metric.
- Use evidence available to the model for detectors; do not introduce hidden
  dependence on labels, graders, or evaluation outcomes. Keep metric definitions,
  provenance, and test examples explicit.
- Keep generated code and imported failure libraries within the documented
  execution boundaries. Preserve the public dashboard's loopback, authentication,
  and request-validation checks. See [SECURITY.md](SECURITY.md).
- Use the terms in the [glossary](docs/glossary.md) in documentation, user interface
  text, and skills, for example *detector* rather than *function* and *failure
  discovery* rather than *detector discovery*. Add a glossary entry when you introduce
  a new user-facing concept. Prompts and rubrics whose text feeds a cache identity,
  such as the behavior-judge prompt and the cheating rubric, change only with a
  deliberate cache migration.
- Avoid unrelated formatting, dependency updates, and generated artifacts. Explain
  new dependencies and any third-party material so maintainers can review their
  provenance and required notices. Do not add dependencies, tools or fixtures that
  install or bundle FFmpeg, PyAV or audio/video codecs beyond those that
  `tools/build_ffmpeg.sh` builds, for example PyPI PyAV wheels or faster-whisper with
  its default PyAV wheel; test media use only that build's codecs. Pillow's AV1 and
  WebP image codecs are reviewed in `third_party/NATIVE_BINARY_REVIEW.md`.

## Validation and review

Run the relevant tests while developing, then the full suite for code changes.
For example, when changing trajectory handling:

```sh
uv run --python 3.12 --extra dev python -m pytest tests/test_trajectory.py
uv run --python 3.12 --extra dev python -m pytest -n auto
uv run --python 3.12 --extra dev ruff check src/autofa/data/trajectory.py
```

Choose the test and lint paths for your actual change, then run the
[quality gates](#quality-gates) before pushing. Add regression tests
for behavior changes and bug fixes, including meaningful failure cases. Avoid tests
that merely repeat the implementation. Documentation-only changes need link,
example, and configuration checks rather than a new test of the prose. Do not
reformat unrelated files; when you remove a ratchet entry, keep that cleanup in
its own commit or pull request.

### Quality gates

CI enforces these gates on the whole tree. Each passes on the code as it was when
the gate was added. Code that existed then and fails a gate is listed as an
exception. The lists may shrink but must not grow: new and changed code meets
every gate. Run the gates locally with:

```sh
uv run --python 3.12 --extra dev ruff check .
uv run --python 3.12 --extra dev python tools/check_layering.py
uv run --python 3.12 --extra dev python tools/check_types.py
uv run --python 3.12 --extra dev python -m pytest -n auto --cov
```

| Gate | What it enforces | Today's exceptions |
| --- | --- | --- |
| Ruff | `E`, `F`, `I`, `UP` and `B` on every Python file, 100-character lines | Files with long lines: `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml` |
| Complexity | Ruff `C901`: cyclomatic complexity of at most 25 per function | `# noqa: C901` on the `def` line of each function over the limit |
| Layering | `autofa.core`, `autofa.data` and `autofa.metrics` do not import `autofa.mcp`, `autofa.ui`, `autofa.cli` or `autofa.session`, including imports deferred inside functions or guarded by `TYPE_CHECKING` | One `importer -> imported` line per module pair in `tools/layering_exceptions.txt` |
| Types | Pyright in basic mode on every file under `src` | `ignore` under `[tool.pyright]` in `pyproject.toml` |
| Coverage | Statement coverage of `src` (`pytest --cov`), reported in the job summary, of at least `fail_under` | The floor in `[tool.coverage.report]` in `pyproject.toml` |

To remove an exception, fix the code and delete its entry in the same change.
Each gate also fails while an exception no longer suppresses anything, so stale
entries cannot linger:

- **Long lines.** Wrap the lines, then delete the file's entry. If the file still
  needs another code, such as `E402`, delete only `E501`. `tests/test_quality_gates.py`
  fails while an entry suppresses no finding.
- **Complexity.** Split the function until Ruff reports a complexity of 25 or less,
  then delete the `# noqa: C901` comment. Ruff's `RUF100` reports the comment as
  unused once the function passes.
- **Layering.** Move the shared code into a lower layer, pass the dependency in as an
  argument, or depend on a `Protocol` defined in the lower layer, then delete the
  line. `tools/check_layering.py` fails while a line matches no import. Run it with
  `--list` to print the current violations.
- **Types.** Fix the file's Pyright errors, then delete its `ignore` entry.
  `tools/check_types.py` runs Pyright with the list removed, and fails if a listed file
  has no errors or an unlisted file has any. Plain `pyright` or an editor shows the
  same result as CI for files outside the list.
- **Coverage.** When a change raises total coverage, you may raise `fail_under` to the
  new total, rounded down to one decimal place. Never lower it; add tests instead.

Adding an exception needs a maintainer's approval and a reason in the pull request.
For example, an update to the bundled evalrdl snapshot can bring long lines from
upstream. Ruff and Pyright are pinned in the `dev` extra because new releases add
checks; upgrade them in a separate change that adjusts the code or the lists.

Use the [pull request template](.github/pull_request_template.md). Include the
problem and resulting behavior, the scope of the
change, validation commands and results, and any compatibility or migration
notes. Update affected documentation and describe user-visible changes in the PR
for inclusion in [GitHub release notes](https://github.com/NVIDIA/AutoFA/releases).
Explain what changed and why in commit messages.

### Pre-commit hooks

[`.pre-commit-config.yaml`](.pre-commit-config.yaml) checks the files in each commit:
YAML, TOML and JSON syntax, merge-conflict markers, files over 1 MB except versioned
`.tar.zst` release data and the two browser-openable example HTML reports under
`eval_data/`, private keys, trailing whitespace and final newlines, Ruff, the layering
contracts, license headers and the
[secrets scan](#secrets-scan). The whitespace hooks skip `eval_data/*.tar.zst`,
`tests/fixtures/` and `third_party/licenses/`, whose files keep their original bytes. Ruff, detect-secrets
and the repository's own checks run through `uv` with the versions pinned in the
`dev` extra. Pyright and the test suite
are not hooks because they take longer; run them as described under
[quality gates](#quality-gates).

```sh
uv run --python 3.12 --extra dev pre-commit run --all-files
```

CI runs every hook on every file. Update the pinned `pre-commit-hooks` revision with
`pre-commit autoupdate` in a separate change.

### Secrets scan

[detect-secrets](https://github.com/Yelp/detect-secrets) checks tracked files for
credentials. Besides its built-in detectors (keywords, high-entropy strings, and
GitHub, GitLab, OpenAI, AWS and other provider tokens),
[`tools/secret_detectors.py`](tools/secret_detectors.py) flags NVIDIA API keys
(`nvapi-…`) and hostnames under `nvidia.com` that are not on its list of public hosts.
Internal service names are thus caught without being written into the repository. Add
a host to `PUBLIC_NVIDIA_HOSTS` only after confirming it is reachable without NVIDIA
network access or credentials.

[`.secrets.baseline`](.secrets.baseline) lists the findings reviewed when the scan was
added: environment-variable names, synthetic test keys, hashes, and URLs used by
redaction tests, all marked as false positives. `tools/check_secrets.py` fails on any
finding the baseline does not list. It matches entries by file, type and hash, so
moving a listed line needs no baseline change. The generated files under
`third_party/` are not scanned. The baseline loads the AutoFA detectors by a
repository-relative path, so run detect-secrets from the repository root. It also
turns off detect-secrets' `is_indirect_reference` filter, which drops every finding
on a line shaped like `client = Client("…")`, including a token or internal URL
passed as an argument. Keep that filter out when you refresh the baseline;
`tests/test_repository_checks.py` fails if it returns.

When the scan reports a finding:

- **A real credential or internal hostname:** remove it, revoke or rotate a credential
  with its provider, and tell a maintainer. A later commit that deletes it does not
  remove it from Git history.
- **A synthetic value:** prefer one the scan does not flag, such as a host under
  `example.com`, or add `# pragma: allowlist secret` to the line. To list it in the
  baseline instead, refresh the baseline, mark the new entry in the audit, and explain
  it in the pull request:

```sh
uv run --python 3.12 --extra dev detect-secrets scan --baseline .secrets.baseline
uv run --python 3.12 --extra dev detect-secrets audit .secrets.baseline
```

### CI and review

The [CI workflow](.github/workflows/ci.yml) runs `Python 3.12 tests and lint` for
pull requests and pushes to `main`. It runs the [quality gates](#quality-gates), the
[pre-commit hooks](#pre-commit-hooks) and the full pytest suite with coverage.

The [install workflow](.github/workflows/install.yml) builds the wheel and installs it
without extras on Python 3.12, 3.13 and 3.14. `tools/check_wheel_install.py` then
requires every source module and `package-data` file in the wheel, imports every
module, and runs `autofa --help`. Its docstring shows how to run it locally.

Workflows pin each action to a full commit SHA with a version comment.
[Dependabot](.github/dependabot.yml) proposes monthly updates to those pins.
[CodeRabbit](.coderabbit.yaml) reviews pull requests with rules that restate this
guide; update both when the coding guidelines change.

Obtain maintainer review and a passing check on the
latest revision before merging. Review and rerun checks after changes that affect
the reviewed behavior. Repository administrators enforce reviews and required
checks through GitHub branch protection.

The [compliance workflow](.github/workflows/compliance.yml) checks source license
headers and packaged legal-file declarations. It also validates author sign-offs
on external PRs. Run the header check locally with
`python tools/check_license_headers.py`; see
[source notices](#source-notices-and-third-party-material) and
[DCO instructions](#developer-certificate-of-origin).

Maintainers review correctness, evidence, compatibility, documentation, and the
release boundary before merging. Maintainers verify the
[license terms](#license-of-contributions), contribution provenance and
[external contributor sign-offs](#developer-certificate-of-origin) before accepting
contributions.
A code review does not replace the applicable legal terms.

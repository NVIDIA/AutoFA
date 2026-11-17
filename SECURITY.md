# Security policy

## Report a vulnerability privately

Use [NVIDIA's vulnerability reporting channels](https://nvidia.com/en-us/security/report-vulnerability/)
or email [psirt@nvidia.com](mailto:psirt@nvidia.com). Encrypt email reports using
the [NVIDIA PSIRT PGP key](https://nvidia.com/en-us/security/pgp-key/).
Do not put vulnerability details or exploit code in public issues, pull requests,
discussions, or logs.

Identify AutoFA and the affected commit or version. Include the operating system,
Python version, relevant configuration, reproduction steps, observed behavior,
and potential impact. Prefer a small synthetic example; remove credentials,
personal information, and private evaluation data from attachments.

The [NVIDIA Product Security portal](https://nvidia.com/en-us/security/) provides
security notices and the applicable disclosure policies. This project does not
promise a separate response deadline or maintenance period.

## Release status

The open research distribution is being prepared for its first public release.
There is no published security support matrix yet. Reports concerning the current
development revision can use the channels above; include the exact commit.

## AutoFA security boundaries

- AutoFA processes evaluation data and sends selected content to the configured
  model providers. Check provider settings and data permissions before a run.
- Python detectors execute inside the AutoFA process with restricted imports and
  builtins. Those restrictions are not operating-system isolation. Review
  imported failure libraries and use a separate environment for untrusted code.
- The Library binds to loopback by default. With `autofa serve --host`, remote
  API and browser access require its shared server token. All token holders use
  the same profile and can modify its libraries; this is not tenant isolation.
  Its token and browser request checks do not isolate other processes running as
  your user. Use a trusted network or a protected transport for remote access.
- Profiles, saved runs, and reports can contain source data, prompts, model
  responses, and credentials. Keep profiles private and inspect artifacts before
  sharing them. Never include the profile token or API keys in a report.

Report suspected failures of these protections through the private channels above.

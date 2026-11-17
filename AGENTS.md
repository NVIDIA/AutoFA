# AutoFA: instructions for the host agent

These setup instructions apply to the outer host agent. If you are executing an
AutoFA-managed discovery, detector or judge task, follow that task's native instructions;
do not bootstrap another driver, install host skills or load the driver into your task.

When the user points you at this checkout to set up or use AutoFA, complete the setup and
continue as the AutoFA driver in the same conversation. Do not stop at installation advice.

1. Read [the driver skill](agent-skills/autofa-driver/SKILL.md) directly and use its instructions
   now, even if `autofa-driver` is not yet in your host's skill catalog.
2. Follow [agent setup](docs/agent-setup.md): install missing prerequisites and this checkout's
   environment, register the outer driver with the current host, and verify the local CLI.
   Reuse a working installation and existing configuration. Resolve paths from this file's
   directory, not an assumed working directory or a different AutoFA checkout.
   Office rendering, audio/video inspection and speech transcription are optional: ask the
   user which they need and install only those, using only the commands in
   [optional readers](docs/agent-setup.md#optional-readers-ask-first). Never install PyAV
   from PyPI or build FFmpeg with codecs outside the documented build.
3. Continue with the user's goal using the driver: establish inputs and constraints, review
   model/effort choices and relevant cloud detectors, then execute the selected native workflow.
   Ask only for missing consequential choices or user-only sign-in/credential actions. If the
   user requested setup alone, report readiness and invite their analysis goal without running one.
   For a discovery campaign, establish the current host's goal and acknowledged monitor before
   launching, including downloads. Review every ten minutes by default and on significant
   events; let healthy native work proceed. Closing this session or pausing monitoring does
   not stop AutoFA. Support live interval changes and later reattachment to the same run.
   Distinguish monitoring pause from an explicit analysis stop. For an ordinary stop request,
   ask whether to finish full-dataset scoring and whether to publish the report, recommending
   yes to both and reusing explicit answers in the stop request. Follow the driver's
   [graceful-stopping workflow](agent-skills/autofa-driver/references/operations.md#graceful-stopping)
   through the chosen report delivery; an immediate stop or refusal of further paid work takes
   priority. A local report or successful launch does not satisfy a cloud-report request.

Setup verification uses help, imports and authentication status; it does not launch an
analysis or publish a report. Loading this file alone does not authorize a global installation.
Preserve user settings, credentials, existing skills and saved analyses. Keep this bootstrap
and the driver outside AutoFA's internal analysis/judge skill context.

For repository development/review requests, follow [CONTRIBUTING.md](CONTRIBUTING.md).
Do not install host skills as a side effect of a code change. Use temporary `AUTOFA_HOME`
profiles and synthetic providers in tests; follow the repository's test and documentation
requirements. Keep unrelated worktree changes intact.

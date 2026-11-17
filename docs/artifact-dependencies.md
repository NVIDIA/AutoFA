# Artifact inspection dependencies

Discovery agents and reviewers can check the current machine and explicitly
prepare a needed reader with `prepare_artifact_support`. A normal artifact read
does not install packages or fetch model files.

```python
prepare_artifact_support(capability="office")  # Inspect availability.
prepare_artifact_support(capability="office", install=True)
prepare_artifact_support(capability="speech", install=True, model="base")
```

The installer accepts only the supported capability names and fixed package
commands. It does not execute installation commands supplied by an artifact.
Already available dependencies are reused. Setup can take several minutes;
cancellation stops the installer, and a timeout or permission failure returns
explicit diagnostics so the agent does not mistake an unread file for missing
evidence.

## Office pages and slides

PDF page rendering, image decoding and native DOCX/XLSX/PPTX text readers are
included in AutoFA's Python dependencies. Office visual rendering additionally
uses [LibreOffice](https://www.libreoffice.org/download/download-libreoffice/).
The setup tool uses an existing platform package manager:

| Platform | Installation command |
| --- | --- |
| macOS with Homebrew | `brew install --cask libreoffice` |
| Debian/Ubuntu | `apt-get install -y libreoffice fonts-dejavu-core` |
| Fedora | `dnf install -y libreoffice` |
| Windows with winget | `winget install --id TheDocumentFoundation.LibreOffice --exact` |

Linux installs use `sudo -n` when necessary and never ask for a password inside
the agent session. If the host requires administrator approval, install the
package outside AutoFA and retry the same artifact view. The tool does not
install a missing package manager.

AutoFA finds LibreOffice on PATH, in standard macOS application folders, and in
Windows Program Files. For a custom installation, set `AUTOFA_LIBREOFFICE` to the
absolute `soffice` executable path. No user profile changes are required.

Conversion uses an isolated LibreOffice profile with macros and external-data
refresh disabled; existing macro/link checks still apply. Page images are a
print-layout conversion, not a replacement for the original document. Read
native cells/text as well when hidden content, notes or formulas matter.
Office fonts affect rendering; install the document's fonts on the host when
exact typography is material to the claim.

## Audio and speech

PyAV supplies audio/video decoding through AutoFA's normal dependencies.
`prepare_artifact_support(capability="audio", install=True)` can repair a missing
PyAV dependency in the active virtual environment.

Speech transcription uses the optional
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) package and explicitly
downloaded `tiny`, `base` or `small` model files. Use the `speech` setup request
above before transcribing. The dependency is installed in AutoFA's active
virtual environment; global Python environments are not modified. Model files
are cached and reused. Ordinary transcription and evidence revalidation use
local files and do not download a model implicitly.

Transcripts are machine-generated evidence, with timestamps and provenance;
they can be inaccurate and do not establish music, sound effects or speaker
identity. Original files remain available for independent inspection.

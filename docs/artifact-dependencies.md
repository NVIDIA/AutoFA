# Artifact inspection dependencies

Discovery agents and reviewers can check the current machine and explicitly
prepare a needed reader with `prepare_artifact_support`. A normal artifact read
does not install packages or fetch model files.

```python
prepare_artifact_support(capability="office")  # Inspect availability.
prepare_artifact_support(capability="office", install=True)  # Install LibreOffice.
prepare_artifact_support(capability="audio")  # Check for PyAV; never installs.
prepare_artifact_support(capability="speech", install=True, model="base")  # Download a model.
```

With `install=True`, the tool installs LibreOffice for `office` and downloads the
selected speech model for `speech`. It never installs Python packages, PyAV or
FFmpeg; `audio` only reports whether PyAV is available. The installer accepts only
the supported capability names and fixed package commands. It does not execute
installation commands supplied by an artifact.
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

## Audio, video and speech

AutoFA declares, installs and distributes no FFmpeg or PyAV, so its standard
installation (`uv sync` or `pip install`) brings in no audio or video decoder; the
only codecs among its dependencies are Pillow's AV1 and WebP image codecs
([notices](../THIRD_PARTY_NOTICES.md#dependency-notice-snapshot)). PyAV's PyPI
wheels bundle an FFmpeg with many more codecs, including the GPL-licensed x264 and
x265 encoders for H.264 and HEVC, so AutoFA no longer installs them. Audio and video
inspection and speech transcription are optional and need a PyAV that you build from
source against an FFmpeg on your host.

Without PyAV, audio and video artifacts stay downloadable and are reported as
unavailable, with a link to this section. An unavailable artifact is an evidence gap,
never evidence that its content is absent.

### Build FFmpeg and PyAV

[`tools/build_ffmpeg.sh`](../tools/build_ffmpeg.sh) downloads FFmpeg 9.0.2, checks its
SHA-256 and builds shared libraries that decode only:

- Video: AV1 (dav1d), VP8 and VP9 (libvpx), and Theora (FFmpeg's own decoder, which
  also registers the VP3 decoder it is built on).
- Audio: Opus (libopus), Vorbis (libvorbis), FLAC and linear PCM.
- Containers: MP4/QuickTime (`mov`), WebM/Matroska, Ogg, FLAC, WAV and AIFF.

Only so that the test suite can write a small video, it also builds the libvpx VP9
encoder, the WebM muxer and the `vp9_superframe` bitstream filter. It enables no other codec, no hardware codec interface and no
network protocol, and needs none of FFmpeg's `--enable-gpl`, `--enable-version3` or
`--enable-nonfree` options. FFmpeg reports the build's license as "LGPL version 2.1
or later".

Install the build prerequisites first:

| Platform | Command |
| --- | --- |
| Ubuntu/Debian | `sudo apt-get install build-essential pkg-config nasm zlib1g-dev python3-dev libdav1d-dev libvpx-dev libopus-dev libvorbis-dev` |
| macOS with Homebrew | `brew install pkgconf dav1d libvpx opus libvorbis`, plus `nasm` on Intel Macs |

`python3-dev` provides the headers PyAV compiles against when the environment uses
the system Python. The build is verified on Ubuntu 24.04 and on macOS with Apple
silicon.

From the AutoFA checkout, after the standard installation, build FFmpeg into an
absolute prefix and PyAV against it. `PYTHON` names the interpreter that
`uv run --no-sync autofa` uses, so the build does not land in another activated
environment:

```sh
PREFIX="$HOME/.local/autofa-ffmpeg"
tools/build_ffmpeg.sh "$PREFIX"
PYTHON="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"
PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" LDFLAGS="-Wl,-rpath,$PREFIX/lib" \
  uv pip install --python "$PYTHON" --no-binary av --no-cache --reinstall-package av \
  "av==19.0.2"
```

PyAV 19.x supports FFmpeg 9.x. `--no-binary av` means the PyPI wheel, with its
bundled FFmpeg, is never used. `--reinstall-package av` replaces a PyAV wheel that an
earlier AutoFA version installed. PyAV loads the libraries from the prefix, so keep
that directory. Check that only the codecs above are available:

```sh
uv run --no-sync python -c "import av; print(sorted(av.codecs_available))"
```

`prepare_artifact_support(capability="audio")` then reports `available`.

### Speech transcription

Speech transcription uses the optional
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) package, which requires
PyAV. Install it after building PyAV, with `PYTHON` set as above; `--no-binary av`
keeps uv from substituting the PyPI wheel for your build:

```sh
uv pip install --python "$PYTHON" --no-binary av "faster-whisper>=1.1,<2"
```

faster-whisper brings CTranslate2, ONNX Runtime and Hugging Face Hub, each with its
own license; you install them. Then use the `speech` setup request above before
transcribing. It reports whether PyAV and faster-whisper are available, never
installs Python packages and, when both are present, downloads the selected `tiny`,
`base` or `small` model (Systran's CTranslate2 conversion of OpenAI Whisper, MIT)
from Hugging Face into AutoFA's cache. Model files are cached and reused. Ordinary
transcription and evidence revalidation use local files and do not download a model
implicitly.

Transcripts are machine-generated evidence, with timestamps and provenance;
they can be inaccurate and do not establish music, sound effects or speaker
identity. Original files remain available for independent inspection.

### Keep the build installed

`uv sync --inexact`, AutoFA's documented installation, keeps PyAV and faster-whisper.
A plain `uv sync` is exact: it removes packages that AutoFA does not declare,
including them. Rerun the commands above if that happens.

### What the documented build reads

AutoFA recognizes the same audio and video file types as before and uses whatever
demuxers and decoders the linked FFmpeg provides. With the documented build:

- Video in MP4/QuickTime and WebM/Matroska files is read when it is AV1, VP8, VP9 or
  Theora. H.264 and HEVC video, used by most MP4 and MOV files, is unavailable.
- Audio in WAV, AIFF, FLAC and Ogg files is read when it is linear PCM, FLAC, Opus
  or Vorbis. AAC audio, used by M4A files, and G.711 mu-law or A-law audio, used by
  many 8 kHz telephony WAV files, are unavailable.
- `.mp3` and `.aac` files are unavailable because the build has no demuxer for them.

A missing demuxer or decoder is reported as unavailable, not as an error: an evidence
gap, never evidence that content is absent. The original file remains downloadable.

You may build FFmpeg with other codecs, for example H.264, HEVC, AAC or MP3, and
build PyAV against it in the same way. The license and patent terms of codecs you
add are yours to review.

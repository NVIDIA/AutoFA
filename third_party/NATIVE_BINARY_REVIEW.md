# Native audio and video codec review

Checked on 2026-11-18. This review covers the audio and video codec libraries in the
declared runtime dependencies of [the dependency notice snapshot](dependency-notices.json),
in the CPython 3.12 Linux x86_64 and aarch64 and macOS arm64 wheels named below.
Inspection was static (`objdump`, `otool`, `nm` and `strings`); the binaries were not
executed. This is not a complete SBOM of every bundled or statically linked library.

## PyAV and the audio extra were removed

AutoFA previously declared PyAV (`av`) as a runtime dependency and faster-whisper as
the optional `audio` extra. The PyAV 19.0.2 PyPI wheels bundle FFmpeg 9.0.2 linked to
libx264 and libx265 (GPL-2.0-or-later), and include LAME and decoders and encoders for
H.264, HEVC, AAC, MP3 and other formats. The previous version of this review
(2026-10-30, rechecked 2026-11-10) found, in both Linux wheels, `DT_NEEDED` entries
from `libavcodec` to the bundled `libx264` and `libx265`, and FFmpeg configuration
strings with `--enable-libx264`, `--enable-libx265` and `--enable-version3` but
without `--enable-gpl`.

AutoFA now declares, installs and distributes no PyAV or FFmpeg; the only codecs in its
declared dependencies are Pillow's, described below. Removing PyAV and faster-whisper also removed CTranslate2 (with its bundled
`libgomp` and its oneMKL/oneDNN or OpenBLAS components), ONNX Runtime and the other
packages only faster-whisper needed: 14 packages in all. Their notices are no longer
retained.

## Pillow bundles AV1 and WebP image codecs

Pillow 13.0.0 was inspected in these wheels; each SHA-256 matched before extraction:

- `pillow-13.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
- `pillow-13.0.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`
- `pillow-13.0.0-cp312-cp312-macosx_11_0_arm64.whl`

The manifest records their hashes. In all three, `libavif` 1.4.2 (for AVIF images) has
no dynamic dependency beyond the C/C++ runtime and contains, linked statically, the
dav1d 1.5.3 AV1 decoder and the libaom 3.14.1 AV1 encoder; no libaom decoder interface
is present. `libwebp`, `libwebpdemux`, `libwebpmux` and `libsharpyuv` handle WebP
images. The dynamic dependencies and strings of the bundled libraries show no FFmpeg,
x264, x265, libvpx, Opus, Vorbis, Theora, FLAC, LAME, OpenH264, libde265, rav1e,
SVT-AV1 or libgav1 library. Pillow's retained [LICENSE](licenses/pillow-13.0.0/LICENSE)
includes the AOM, DAV1D, LIBAVIF and LIBWEBP notices. The retained notices of the other
declared dependencies name no audio or video codec library.

## Optional FFmpeg built by the user

Audio and video inspection is optional and uses a PyAV that the user builds from
source against an FFmpeg on their host ([setup](../docs/artifact-dependencies.md#audio-video-and-speech)).
The repository distributes the build script `tools/build_ffmpeg.sh`, not binaries. It
builds FFmpeg 9.0.2 shared libraries that decode only AV1 (dav1d), VP8 and VP9
(libvpx), Theora (FFmpeg's decoder, which also registers the VP3 decoder it is built
on), Opus (libopus), Vorbis (libvorbis), FLAC and linear PCM. The libvpx VP9 encoder,
the WebM muxer and the `vp9_superframe` bitstream filter exist only so the test suite
can write a small video. The build enables no
other codec, no hardware codec interface and no network protocol, and uses none of
`--enable-gpl`, `--enable-version3` or `--enable-nonfree`; FFmpeg reports its license
as "LGPL version 2.1 or later". CI checks that the PyAV built against it reports
exactly this codec list. The license and patent terms of codecs that a user adds are
theirs to review.

## Redistribution

A container image, copied virtual environment or other bundle that includes installed
dependency binaries redistributes them and needs its own notices and review.

# Native dependency license verification

This records the technical findings for PyAV 19.0.2 and CTranslate2 4.8.4 in
the CPython 3.12 Linux x86_64 and aarch64 wheels inspected on 2026-10-30.
[The dependency manifest](dependency-notices.json) records the wheel hashes,
individual library hashes, ELF dependencies, build strings, source references,
and retained notice-file hashes. Inspection was static; the binaries were not
executed.

The evidence was re-verified on 2026-11-10 from the same PyPI wheels. Wheel and
library hashes, `DT_NEEDED` entries, build strings, and the `libgomp` GNU build IDs
and `.text`/`.rodata` hashes all matched the manifest. CTranslate2 4.8.5, the
release resolved on that date, bundles the identical `libgomp` binaries.

## Distribution scope

AutoFA's wheel contains its Python code and resources and declares dependency
requirements. Inspection of the built wheel found no bundled `.so`, `.dylib`,
`.dll`, or dependency wheel files. A normal installation obtains dependencies
separately. CTranslate2 is brought in by the optional `audio` extra through
faster-whisper; PyAV is a declared runtime dependency used for audio and video.

Publishing a container image, copied virtual environment, or other bundle that
contains installed dependencies would redistribute their native binaries.
That distribution needs the applicable component notices and source-compliance
arrangements. The findings below also belong in the dependency review for
AutoFA's declared-dependency distribution model.

## CTranslate2's libgomp: license and provenance verified

Both CTranslate2 wheels contain and link `libgomp`, the GNU OpenMP runtime.
The bundled files match the GNU build IDs and byte-for-byte `.text` and `.rodata`
sections of AlmaLinux's `libgomp-8.5.0-28.el8_10.alma.1` packages for their
respective architectures. Wheel bundling changes the filename and dynamic
linking metadata; this comparison does not claim identical whole-file hashes.

The RPMs identify `gcc-8.5.0-28.el8_10.alma.1.src.rpm` as their source package.
That source RPM was downloaded and hashed, and its GCC source archive supplied
the retained copyright notice, GPLv3 text, and GCC Runtime Library Exception 3.1:

- [Copyright and license notice](licenses/ctranslate2-4.8.4/native/libgomp-8.5.0/NOTICE)
- [GPLv3](licenses/ctranslate2-4.8.4/native/libgomp-8.5.0/COPYING3)
- [GCC Runtime Library Exception 3.1](licenses/ctranslate2-4.8.4/native/libgomp-8.5.0/COPYING.RUNTIME)

The identified license is **GPL-3.0-or-later WITH GCC-exception-3.1**.
The exception permits qualifying independent modules to use other licenses;
`libgomp`'s presence alone does not require changing AutoFA's license to GPL.
The exception has conditions and does not remove the obligations on a
redistributed copy of the runtime itself. Its exact source-RPM URL and checksum
are recorded in the manifest for preparation of any binary-distribution source
bundle. See the [GNU explanation](https://www.gnu.org/licenses/gcc-exception-3.1-faq.html.en).

## PyAV's FFmpeg/codecs: GPL-component finding verified

The PyAV Python bindings carry a BSD-3-Clause notice. The inspected Linux wheels
also contain FFmpeg and codec libraries with their own terms:

- In both architectures, ELF `DT_NEEDED` entries show that `libavcodec` links the
  bundled `libx264` and `libx265` libraries. This is a linked dependency, beyond
  simply finding unused files in an archive.
- The FFmpeg configuration strings contain `--enable-version3`,
  `--enable-libx264`, and `--enable-libx265`, and omit `--enable-gpl`.
- PyAV's [pinned build recipe](https://github.com/PyAV-Org/pyav-ffmpeg/blob/9.0.2-1/scripts/pkg.py)
  identifies FFmpeg 9.0.2, x264 revision
  `b35605ace3ddf7c1a5d67a2eb553f034aef41d55`, and x265 4.3.
  The x265 binaries report `4.3+1-e9b8812`.
- That build's [FFmpeg configuration patch](https://github.com/PyAV-Org/pyav-ffmpeg/blob/9.0.2-1/patches/ffmpeg.patch)
  moves x264 and x265 from the GPL external-library list to the version-3 list.
  This explains why the flag alone is insufficient evidence of LGPL-only terms.

The inspected x264 and x265 source notices specify **GPL-2.0-or-later** and
mention separately available commercial licensing. No alternative commercial
grant was supplied in the inspected wheel notices. FFmpeg's own
[license guidance](https://github.com/FFmpeg/FFmpeg/blob/n9.0.2/LICENSE.md#external-libraries)
lists these codecs as GPL libraries and states that combining them with FFmpeg
requires GPL licensing for the combination. Changing a build-check list does
not itself supply different copyright permissions.

The retained evidence includes [FFmpeg's license guide and license texts](licenses/av-19.0.2/native/ffmpeg-9.0.2/),
[x264's notice and GPL text](licenses/av-19.0.2/native/x264-b35605a/), and
[x265's notice and GPL text](licenses/av-19.0.2/native/x265-4.3/).
These wheels contain GPL-licensed components; package-level BSD metadata does
not describe all terms applicable to the binary bundle.

## Redistribution considerations and review scope

The applicable obligations depend on the selected binaries and distribution
model. A build without these codecs or one covered by documented alternative
licensing can have different terms. This technical review does not replace
AutoFA's source-license terms or the individual dependency licenses.

If dependency binaries or container images will be distributed, prepare the
applicable notices and corresponding-source arrangements for the selected
artifacts. A source URL in this manifest is evidence for that work, not by itself
completion of every binary-distribution obligation.

This targeted review establishes the libgomp provenance/license and the
FFmpeg/x264/x265 finding. It is not a complete SBOM of all PyAV system libraries
or all statically linked CTranslate2 components. For example, CTranslate2's
[release build recipe](https://github.com/OpenNMT/CTranslate2/blob/v4.8.4/python/tools/prepare_build_environment_linux.sh)
also selects architecture-specific OpenBLAS or MKL/oneDNN components. Complete
that inventory for any distribution that includes those binaries.

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build FFmpeg's shared libraries with only the AV1, VP8, VP9 and Theora video codecs and
# the Opus, Vorbis, FLAC and linear PCM audio codecs, so PyAV can be compiled against them.
# Install the codec libraries first; "Audio, video and speech" in
# docs/artifact-dependencies.md lists them and shows how to build PyAV against the result.
#
#   tools/build_ffmpeg.sh ABSOLUTE_PREFIX
#
# AV1 is decoded by dav1d, VP8 and VP9 by libvpx, Opus by libopus and Vorbis by libvorbis.
# Theora, FLAC and PCM use FFmpeg's own decoders; its Theora decoder also registers the VP3
# decoder it is built on. The libvpx VP9 encoder, the WebM muxer and the VP9 bitstream
# filter exist only so the test suite can write a small video. The build enables no other
# codec, no hardware codec interface and no network protocol, and needs none of FFmpeg's
# --enable-gpl, --enable-version3 or --enable-nonfree options.
set -euo pipefail

version=9.0.2
sha256=8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e

if [[ $# -ne 1 || $1 != /* ]]; then
  echo "usage: $0 ABSOLUTE_PREFIX" >&2
  exit 2
fi
prefix=$1

pcm=pcm_u8,pcm_s8,pcm_s16le,pcm_s16be,pcm_s24le,pcm_s24be,pcm_s32le,pcm_s32be,pcm_s64le
pcm=$pcm,pcm_f32le,pcm_f32be,pcm_f64le,pcm_f64be

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
archive=$work/ffmpeg-$version.tar.xz
curl -fsSL --retry 3 -o "$archive" "https://ffmpeg.org/releases/ffmpeg-$version.tar.xz"
if command -v sha256sum >/dev/null; then
  actual=$(sha256sum "$archive")
else
  actual=$(shasum -a 256 "$archive")
fi
if [[ ${actual%% *} != "$sha256" ]]; then
  echo "ffmpeg-$version.tar.xz does not match its recorded SHA-256" >&2
  exit 1
fi
tar -xf "$archive" -C "$work"
cd "$work/ffmpeg-$version"

# PyAV rejects linker flags in pkg-config output, so the run path that lets the libraries
# find each other goes in --extra-ldflags rather than FFmpeg's --enable-rpath.
./configure \
  --prefix="$prefix" \
  --enable-shared --disable-static \
  --extra-ldflags="-Wl,-rpath,$prefix/lib" \
  --disable-programs --disable-doc --disable-network \
  --disable-autodetect --disable-iconv --enable-zlib \
  --disable-everything \
  --enable-libdav1d --enable-libvpx --enable-libopus --enable-libvorbis \
  --enable-decoder="libdav1d,libvpx_vp8,libvpx_vp9,theora,libopus,libvorbis,flac,$pcm" \
  --enable-encoder=libvpx_vp9 \
  --enable-parser=av1,vp8,vp9,vp3,opus,vorbis,flac \
  --enable-demuxer=mov,matroska,ogg,flac,wav,aiff \
  --enable-muxer=webm \
  --enable-bsf=vp9_superframe \
  --enable-protocol=file \
  --enable-filter=aformat,aresample \
  || { echo "configure failed; end of ffbuild/config.log:" >&2; tail -n 80 ffbuild/config.log >&2
       exit 1; }
make -j"$(getconf _NPROCESSORS_ONLN)"
make install

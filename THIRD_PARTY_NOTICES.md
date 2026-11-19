# Third-party notices

AutoFA uses the third-party packages below. Their copyright, license and
attribution texts are copied without modification into `third_party/licenses/`.
AutoFA also includes the adapted [failure taxonomy](#failure-taxonomy) described below.
The project license does not replace those terms. NVIDIA-authored bundled EvalRDL
components are described in [their component notice](third_party/evalrdl/README.md)
and are covered by AutoFA's source-file and top-level license notices.

## Failure taxonomy

The built-in taxonomies `src/autofa/taxonomy/versions/taxonomy_v0.json` and
`src/autofa/taxonomy/versions/taxonomy_v1.json` are adapted from the Appendix B
failure-mode definitions and Table 1 component vocabulary of
[*Model or Harness? An Interaction-Centric Taxonomy for Localizing Agent Failures*](https://arxiv.org/abs/2607.28802)
by Harsh Raj, Vipul Gupta, Anas Mahmoud, Razvan-Gabriel Dumitru, Darvin Yi,
Aakash Sabharwal and Yunzhong He (Scale AI), arXiv:2607.28802. The paper is
licensed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/)
(CC-BY-4.0), whose text is also included in [LICENSE](LICENSE).

AutoFA modified the material. `taxonomy_v0.json` restates the paper's failure-mode
definitions and repeats their interaction and component context at each failure
mode. `taxonomy_v1.json` further refines the failure modes for multimodal evidence,
reasoning failures and grader errors. The `source` block in each file credits the
paper and describes its adaptation. Taxonomy classification reports, including the
example reports in `eval_data/example_reports/`, reproduce these definitions and
credit the paper.

## Dependency notice snapshot

This inventory records the runtime dependency resolution on **2026-10-30**. It
includes platform-conditional Python packages and excludes development-only tools.
[The manifest](third_party/dependency-notices.json) records package versions,
source paths and SHA-256 hashes for every copied notice.

AutoFA's wheel declares these dependencies; it does not vendor their Python
packages or native libraries. The versions listed here describe this notice
snapshot, not constraints on all future installations.

Native-library notices were collected from the macOS arm64 distributions and
checked against **32 Linux wheels for 16 platform-specific packages**, covering
CPython 3.12 on glibc Linux **x86_64 and aarch64 (ARM64)**. The manifest records
each Linux wheel's filename, download URL, verified SHA-256 and mapping from its
notice files to the retained copies. Identical text shares a copy; Linux-specific
NumPy and PDFium texts are retained in their respective `linux/` notice directories.
This records the notices supplied by the selected wheels, not a complete review
of every library embedded in them.

AutoFA declares no PyAV, FFmpeg or other audio/video codec dependency. Its
declared dependencies do include image codecs: the
[native codec review](third_party/NATIVE_BINARY_REVIEW.md) describes Pillow's AVIF
and WebP codecs (libavif with the dav1d AV1 decoder and the libaom AV1 encoder, and
libwebp), and Pillow and pypdfium2 also read formats such as JPEG, PNG, GIF, TIFF and
JPEG 2000. Audio and video inspection uses a PyAV that the user builds with the
documented FFmpeg build ([setup](docs/artifact-dependencies.md#audio-video-and-speech));
that build is outside this inventory. Review the applicable license terms,
remaining embedded/static libraries and source-compliance arrangements before
redistributing dependency binaries or container images.

Other Python versions, musl/Alpine or custom builds, independently installed model
weights, provider services, and separately redistributed dependency binaries
require their applicable notices and review.
Refresh the inventory when dependency versions or distribution targets change.

| Package | Version | License and attribution files |
| --- | --- | --- |
| annotated-doc | 0.0.5 | [Notices](third_party/licenses/annotated-doc-0.0.5/) |
| annotated-types | 0.8.0 | [Notices](third_party/licenses/annotated-types-0.8.0/) |
| anyio | 4.15.1 | [Notices](third_party/licenses/anyio-4.15.1/) |
| attrs | 26.1.0 | [Notices](third_party/licenses/attrs-26.1.0/) |
| cffi | 2.1.1 | [Notices](third_party/licenses/cffi-2.1.1/) |
| claude-agent-sdk | 0.2.175 | [Notices](third_party/licenses/claude-agent-sdk-0.2.175/) |
| click | 8.5.0 | [Notices](third_party/licenses/click-8.5.0/) |
| cryptography | 50.0.2 | [Notices](third_party/licenses/cryptography-50.0.2/) |
| fastapi | 0.162.0 | [Notices](third_party/licenses/fastapi-0.162.0/) |
| h11 | 0.16.0 | [Notices](third_party/licenses/h11-0.16.0/) |
| httpcore2 | 2.16.0 | [Notices](third_party/licenses/httpcore2-2.16.0/) |
| httpx2 | 2.16.0 | [Notices](third_party/licenses/httpx2-2.16.0/) |
| httpx2-jsfetch | 1.0 | [Notices](third_party/licenses/httpx2-jsfetch-1.0/) |
| idna | 3.20 | [Notices](third_party/licenses/idna-3.20/) |
| ijson | 3.6.0 | [Notices](third_party/licenses/ijson-3.6.0/) |
| jaraco-classes | 3.4.0 | [Notices](third_party/licenses/jaraco-classes-3.4.0/) |
| jaraco-context | 6.1.2 | [Notices](third_party/licenses/jaraco-context-6.1.2/) |
| jaraco-functools | 4.6.0 | [Notices](third_party/licenses/jaraco-functools-4.6.0/) |
| jeepney | 0.9.0 | [Notices](third_party/licenses/jeepney-0.9.0/) |
| jinja2 | 3.1.6 | [Notices](third_party/licenses/jinja2-3.1.6/) |
| jiter | 0.18.0 | [Notices](third_party/licenses/jiter-0.18.0/) |
| jsonschema | 4.26.0 | [Notices](third_party/licenses/jsonschema-4.26.0/) |
| jsonschema-specifications | 2025.9.1 | [Notices](third_party/licenses/jsonschema-specifications-2025.9.1/) |
| keyring | 25.7.0 | [Notices](third_party/licenses/keyring-25.7.0/) |
| markdown-it-py | 4.2.0 | [Notices](third_party/licenses/markdown-it-py-4.2.0/) |
| markupsafe | 3.0.4 | [Notices](third_party/licenses/markupsafe-3.0.4/) |
| mcp | 2.3.0 | [Notices](third_party/licenses/mcp-2.3.0/) |
| mcp-types | 2.3.0 | [Notices](third_party/licenses/mcp-types-2.3.0/) |
| mdurl | 0.1.2 | [Notices](third_party/licenses/mdurl-0.1.2/) |
| more-itertools | 11.2.1 | [Notices](third_party/licenses/more-itertools-11.2.1/) |
| numpy | 2.5.4 | [Notices](third_party/licenses/numpy-2.5.4/) |
| openai | 3.41.0 | [Notices](third_party/licenses/openai-3.41.0/) |
| opentelemetry-api | 1.45.2 | [Notices](third_party/licenses/opentelemetry-api-1.45.2/) |
| pillow | 13.0.0 | [Notices](third_party/licenses/pillow-13.0.0/) |
| polars | 2.1.0 | [Notices](third_party/licenses/polars-2.1.0/) |
| polars-runtime-32 | 2.1.0 | [Notices](third_party/licenses/polars-runtime-32-2.1.0/) |
| prompt-toolkit | 3.0.53 | [Notices](third_party/licenses/prompt-toolkit-3.0.53/) |
| pyarrow | 26.0.0 | [Notices](third_party/licenses/pyarrow-26.0.0/) |
| pycparser | 3.12 | [Notices](third_party/licenses/pycparser-3.12/) |
| pydantic | 2.14.2 | [Notices](third_party/licenses/pydantic-2.14.2/) |
| pydantic-core | 2.50.2 | [Notices](third_party/licenses/pydantic-core-2.50.2/) |
| pygments | 2.21.0 | [Notices](third_party/licenses/pygments-2.21.0/) |
| pyjwt | 2.15.1 | [Notices](third_party/licenses/pyjwt-2.15.1/) |
| pypdfium2 | 5.14.0 | [Notices](third_party/licenses/pypdfium2-5.14.0/) |
| python-multipart | 0.0.32 | [Notices](third_party/licenses/python-multipart-0.0.32/) |
| pywin32 | 312 | [Notices](third_party/licenses/pywin32-312/) |
| pywin32-ctypes | 0.2.3 | [Notices](third_party/licenses/pywin32-ctypes-0.2.3/) |
| pyyaml | 6.0.3 | [Notices](third_party/licenses/pyyaml-6.0.3/) |
| referencing | 0.37.0 | [Notices](third_party/licenses/referencing-0.37.0/) |
| regex | 2026.9.29 | [Notices](third_party/licenses/regex-2026.9.29/) |
| rich | 15.0.0 | [Notices](third_party/licenses/rich-15.0.0/) |
| rpds-py | 2026.9.1 | [Notices](third_party/licenses/rpds-py-2026.9.1/) |
| secretstorage | 3.5.0 | [Notices](third_party/licenses/secretstorage-3.5.0/) |
| sniffio | 1.3.1 | [Notices](third_party/licenses/sniffio-1.3.1/) |
| sse-starlette | 3.6.0 | [Notices](third_party/licenses/sse-starlette-3.6.0/) |
| starlette | 1.10.0 | [Notices](third_party/licenses/starlette-1.10.0/) |
| tomlkit | 0.15.2 | [Notices](third_party/licenses/tomlkit-0.15.2/) |
| truststore | 0.10.5 | [Notices](third_party/licenses/truststore-0.10.5/) |
| typing-extensions | 4.16.0 | [Notices](third_party/licenses/typing-extensions-4.16.0/) |
| typing-inspection | 0.4.4 | [Notices](third_party/licenses/typing-inspection-0.4.4/) |
| uvicorn | 0.54.0 | [Notices](third_party/licenses/uvicorn-0.54.0/) |
| wcwidth | 0.9.4 | [Notices](third_party/licenses/wcwidth-0.9.4/) |

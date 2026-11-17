# Bundled evaluation record libraries

AutoFA bundles selected source files from `evalrdl-schema`, `evalrdl-readers`,
`evalrdl-export`, and `evalrdl-trace` in its own distribution. Their Python imports
retain the upstream package names. Installation does not fetch a separate source
repository or require a hosted evaluation service.

These files are synced from the EvalRDL repository. The current sync is from EvalRDL
revision `53e689376888ae540c7e03786a6b5c552a04c211` (package versions 0.1.0).
Schema versions 11 through 17 are included. The readers include the local importer
(`convert` and `write_tables`); there is no separate importer package.

Runtime dependencies beyond Python are PyYAML (readers), Polars (tables and
exporters), and Starlette/Uvicorn (HTTP hosting). AutoFA declares these through its
package dependencies. Six original upstream test modules accompany the copied
components: `tests/vendor/evalrdl/readers/tests/convert/test_rollout_identity.py`
for the readers, and the five `tests/vendor/test_evalrdl_*.py` modules for the
schema, export and trace. No upstream fixtures or example logs are bundled. The
optional native reader accelerator is not bundled; the Python implementation
remains available without it.

The bundled viewer supplies embedding hooks, bounded full-text reads, stable
part/lane links, text highlighting and theme variables.

The bundled NVIDIA-authored code-only files carry Apache-2.0 source headers;
documentation follows AutoFA's top-level dual-license terms. The full terms are
distributed in [LICENSE](../../LICENSE), with attribution in [NOTICE](../../NOTICE).
The AutoFA integration adds release license headers to the synced source files.
Third-party dependency notices are in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
These components are included in AutoFA's single distribution rather than
published as standalone packages.

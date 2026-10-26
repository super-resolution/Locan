# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

- `locan` is a Python library for analyzing single-molecule localization microscopy (SMLM)
  data: fluorescence super-resolution data consisting of localization coordinates plus
  localization properties (frame, intensity, uncertainty, ...).
- The central object is `LocData` (`src/locan/data/locdata.py`): a pandas DataFrame of
  localizations, protobuf metadata (`meta`), lazily computed hulls, and an optional `Region`.
- Docs: https://locan.readthedocs.io. Development rules: `docs/source/development.rst`.
- Supported Python: `>=3.11` (`requires-python`). Local development pins `3.13` in
  `.python-version`, matching CI. Do not use 3.14 locally: `pytest-qt` (group `test-qt`) is
  capped at 3.13 and `ray` (group `docs`) at 3.12 on Windows.
- Public API follows semantic versioning (see "Public API" below).

## Tooling — always use these, never alternatives

- **uv** manages Python, the venv and dependencies locally. Never call `pip`, `python -m venv`,
  `conda` or bare `python`; prefix everything with `uv run`. `uv.lock` is local and gitignored:
  change dependencies only in `pyproject.toml` via `uv add` / `uv add --dev` / `uv remove`.
  CI and the published install paths deliberately use pip and conda; do not change that.
- **black** is the only formatter. Never run `ruff format`.
- **ruff** is the only linter, including import sorting (no isort, flake8, pylint).
  Black and ruff share `line-length = 88` in `pyproject.toml`; keep them in sync.
- **pyrefly** is the only type checker (`preset = "strict"`).
- **pytest** is the test runner; **coverage** measures coverage (CI gate: 90 %).

## Commands

```bash
uv sync --extra all --extra pyside6       # full dev env: dev group + all optional deps + one Qt binding
uv run pytest                             # extended suite (gui/visual tests deselected by default)
uv run pytest tests/analysis/test_drift.py::test_drift   # single test — prefer while iterating
uv run pytest -x --lf                     # stop on first failure, rerun last failures
uv run pytest -m "not slow and not gui and not visual"   # skip slow tests (-m replaces the default selection)
uv run pytest -m network tests/test_datasets.py          # dataset downloads (deselected by default)
uv run coverage run -m pytest && uv run coverage report   # local coverage (CI requires >= 90 %)
uv run ruff check --fix .                 # lint + autofix (incl. import sorting)
uv run black .                            # format (also code cells in notebooks: black[jupyter])
uv run pyrefly check                      # type check
uv run pre-commit run --all-files         # what the git hooks would run
uv sync --group docs && uv run sphinx-build -b html docs docs/_build/html   # docs (slow: executes all notebooks)
```

## Definition of done

Before reporting a task as finished, run in this order and make all pass:

```bash
uv run ruff check --fix . && uv run black . && uv run pyrefly check && uv run pytest
```

Then add an entry to `CHANGES.rst` under the current dev section (see "Changelog").
If something fails that is unrelated to your change, say so explicitly instead of fixing it silently.

## Layout

```
src/locan/                  # import as `locan`
├── __init__.py             # lazy_loader entry point (see "Public API")
├── __init__.pyi            # the public API stub: every exported name, one explicit re-export each
├── configuration.py        # pydantic-settings `Configuration`, exposed as `locan.config`
├── constants.py            # PropertyKey/PROPERTY_KEYS, file-format key maps, enums (FileType, HullType, RenderEngine)
├── dependencies.py         # optional-dependency machinery: HAS_DEPENDENCY, needs_package, QtBindings
├── datasets.py             # example datasets (download via httpx)
├── locan_types.py          # Protocols for DataFrame / LocData typing
├── analysis/               # analysis classes, all subclasses of `_Analysis` (analysis_base.py)
├── data/                   # LocData, Image, metadata (*.proto, *_pb2.py), regions/, hulls/, adapter/
├── locan_io/               # file IO: files.py, locdata/<format>_io.py, load_locdata dispatcher, manifest
├── process/                # aggregate (Bins, histogram), filter, cluster/, register, tracking, properties/, transform/
├── rois/                   # Roi (region + locdata reference, yaml/toml IO)
├── simulation/             # make_*/simulate_* point processes, drift, tracks
├── visualize/              # render_2d/3d dispatch on RenderEngine → render_mpl/, render_napari/; colormaps
├── gui/                    # Qt file dialogs
├── utils/                  # format, rotation, statistics, system_information
├── scripts/                # CLI subcommands wired in __main__.py (`locan` console script)
└── tests/                  # minimal in-package smoke suite + `locan.tests.test()` runner
tests/                      # extended suite, mirrors the subpackages; fixtures in tests/conftest.py
tests/test_data/            # sample files for every supported format, asdf datasets, legacy metadata
docs/                       # Sphinx (furo, autosummary, myst-nb); docs/tutorials/notebooks/*.ipynb
benchmarks/                 # asv benchmarks
tools/                      # maintainer scripts, e.g. script_make_init.py
.claude/                    # Claude Code settings, hooks, skills (committed)
.github/                    # GitHub templates and workflows for GitHub Actions
```

Large modules: `data/regions/region.py` (~3500 lines), `simulation/simulate_locdata.py`,
`process/aggregate.py`, `data/locdata.py`. Grep first and read by line range; do not load whole.

## Architecture rules

### Public API
- `locan/__init__.py` uses `lazy_loader.attach_stub`; the exported names are whatever
  `locan/__init__.pyi` re-exports. `__all__` is derived from the stub.
- Every module declares `__all__: list[str]`. Each subpackage `__init__.py` star-imports its
  submodules, lists them in `submodules`, builds `__all__` from them, and lists them in an
  `.. autosummary::` block in the module docstring (that block drives the API docs).
- To add a public name: add it to the module `__all__`; if it is a new module, add it to the
  subpackage `__init__.py` (import, `submodules`, autosummary); add an explicit
  `Name as Name` re-export to `locan/__init__.pyi` (`uv run python tools/script_make_init.py`
  prints the blocks to paste). `tests/test_locan_init.py` checks `dir(locan)` against `__all__`.
- Anything exported is a contract. Don't rename/remove it or change its signature incompatibly
  without a deprecation: `warnings.warn("x is deprecated, use y", DeprecationWarning, stacklevel=2)`,
  typically via a module-level `__getattr__` (see `visualize/colormap.py`). Deprecated names must
  not be touched at import time: importing `locan` must emit no `DeprecationWarning` from `locan.*`.

### Analysis classes (`src/locan/analysis/`)
Every analysis is a subclass of `_Analysis` (`analysis_base.py`). Follow an existing one such as
`localizations_per_frame.py`:
- `__init__(self, meta=None, <params>)` → `parameters = self._get_parameters(locals())` →
  `super().__init__(**parameters)`; `self.results = None`.
- A pure, private module-level helper `_<name>(...)` does the computation.
- `compute(self, locdata) -> Self` sets `self.results` (a `_Results` dataclass or DataFrame) and
  returns `self`. Empty `locdata` must be handled (log a warning, leave `results` None).
- Optional: `plot(ax=None, **kwargs) -> Axes`, `hist(...)`, `fit_distributions(...)`, `report()`.
- Metadata is `AMetadata` (protobuf `metadata_analysis.proto`); parameters are recorded automatically.

### Optional dependencies
- Declared as extras in `pyproject.toml`; detected centrally in `dependencies.py` (`HAS_DEPENDENCY`).
- Module pattern: `if HAS_DEPENDENCY["napari"]: import napari` at the top, and
  `@needs_package("napari")` on every public function that needs it (raises `ImportError`).
- Tests: `pytest.importorskip("napari")` at module level or
  `@pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="requires open3d")`.
- The Qt binding is chosen via `config.qt_binding` / `QT_API` before qtpy is imported; never
  import a Qt binding directly.
- Ask before adding a runtime dependency; dev/test tools go in dependency groups.

### Data model
- Localization properties are named by `constants.PropertyKey` (`position_x`, `frame`,
  `intensity`, ...); new properties go there with a `PropertyDescription`.
- Use reverse-style notation for aggregated names, like `frame_max` or `intensity_counts` or `frame_max_mean`.
- Metadata is protobuf. Sources are `data/metadata.proto`, `analysis/metadata_analysis.proto`,
  `locan_io/locdata/manifest.proto`. The generated `*_pb2.py` / `*_pb2.pyi` are checked in and
  excluded from ruff, pyrefly and coverage. **Never edit generated files.** A `.proto` change
  requires regenerating with the latest protoc version.
- `Region` hierarchy in `data/regions/region.py` (shapely-backed 2D, open3d-backed 3D);
  `Roi` wraps a region plus a locdata reference.
- `src/locan/_version.py` is written by setuptools_scm; never edit or commit it.

## Code style

- Full type hints on all functions, methods and class attributes, including `-> None`.
- Modern syntax: `list[int]`, `X | None`, `from collections.abc import Iterable, Sequence, Mapping`.
- Import aliases are enforced by ruff: `np`, `pd`, `plt`, `mpl`, `nx`, `sp`, `bh`, `o3d`.
- Use `n_something` for "number of something". `pathlib.Path`, f-strings,
  `logger = logging.getLogger(__name__)`. No `print` in library code (`report()`/`print_*`
  methods are the documented exceptions).
- Code lines ≤ 88 chars (black); docstring lines ≤ 75 chars. NumPy-style docstrings on every
  function, class and module with `Parameters`, `Returns`, `Raises`, `Attributes`; a short
  `Examples` block on public functions where useful (doctest format).
- No mutable default arguments; no bare `except:`; raise specific exceptions with clear messages.
- Don't add comments that restate the code; explain why. Don't reformat or refactor unrelated code.

## Typing rules

- pyrefly strict must stay clean. New or edited lines suppress errors only with a scoped code and
  reason: `# pyrefly: ignore[bad-assignment]  # reason: upstream stub is wrong`.
- The code base still contains hundreds of bare `# type: ignore` comments from the mypy era.
  They are legacy: leave them alone unless you are already changing that line.
- Never weaken `[tool.pyrefly]` or `[tool.ruff]` in `pyproject.toml` to make errors disappear.
  Same for `# noqa`: always with a rule code, e.g. `# noqa: S603`.
- Avoid `Any`; narrow it immediately and explain in a comment if unavoidable.

## Testing

- Two suites, both in `testpaths`: `src/locan/tests` is a minimal smoke suite shipped with the
  package (`locan test` CLI); `tests/` is the extended suite and mirrors the subpackages.
  New tests go in `tests/<subpackage>/test_<module>.py`.
- Markers (`--strict-markers`): `gui`, `visual` and `network` are deselected by default
  (`addopts`); `slow` is opt-out via `-m "not slow and not gui and not visual"` (a `-m` on the
  command line replaces the one in `addopts`). Mark anything that opens a window, needs a human
  or downloads data. `pytest-timeout` kills any test running longer than `timeout` seconds.
- Reuse the session fixtures in `tests/conftest.py` (`locdata_2d`, `locdata_3d`,
  `locdata_blobs_2d`, `locdata_dSTORM_2d`, `locdata_two_cluster_2d`, ...). They are session-scoped:
  never mutate them; use `deepcopy(locdata)` instead.
- Test data lives in `tests/test_data`; reach it via `from tests import TEST_DIR`.
- Every bug fix gets a regression test that fails before the fix. Every new public function gets
  tests for normal, edge and error cases (`pytest.raises(..., match=...)`).
  Use `@pytest.mark.parametrize` instead of loops; `tmp_path` for files; `monkeypatch` for env.
- Tests are deterministic and offline: seed numpy (`rng = np.random.default_rng(seed)`), no network
  (dataset downloads are marked `network` and deselected by default), no real clock.
- Don't delete or loosen an existing test to make it pass. If a test seems wrong, say why and ask.

## Docs and changelog

- API docs are generated by autosummary from module docstrings; keep the `.. autosummary::`
  lists in `__init__.py` docstrings current when adding modules.
- Tutorials are Jupyter notebooks in `docs/tutorials/notebooks/`, executed by the docs build
  (`nb_execution_mode = "force"`). Do not execute or re-save notebooks unless asked; `black .`
  formatting their code cells is intended. If changed, always save with cleared output.
- **Changelog**: `CHANGES.rst` (reStructuredText, not Keep a Changelog). Add a one-line entry under
  the top `<version> - dev` section in the fitting subsection: `New Features`, `API Changes`,
  `Bug Fixes` (prefixed `fix: ...`), or `Other Changes and Additions`.
- Releases: finalize the `CHANGES.rst` heading with the date, update `version`/`date-released` in
  `CITATION.cff`, tag, and publish a GitHub release; `deploy_to_pypi.yml` uploads to PyPI.

## Workflow

- Read the relevant module and its tests before changing anything.
- For non-trivial changes, outline the plan first; keep diffs small and focused.
- Commits: imperative subject following Conventional Commits (`fix: ...`, `feat: ...`,
  `refactor: ...`), body explains why. Never commit unless asked.
- Do not touch: `*_pb2.py*`, `src/locan/_version.py`, `uv.lock`, `docs/tutorials/notebooks/*.ipynb`,
  `.github/workflows/*` (except when the task is about CI).

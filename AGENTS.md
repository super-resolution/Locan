# AGENTS.md

## Project overview

- `locan` is a Python library with a command-line interface for analyzing single-molecule
  localization microscopy (SMLM) data: fluorescence super-resolution data consisting of
  localization coordinates plus localization properties (frame, intensity, uncertainty, ...).
- The central object is `LocData` (`src/locan/data/locdata.py`): a pandas DataFrame of
  localizations, protobuf metadata (`meta`), lazily computed hulls, and an optional `Region`.
- Public API follows semantic versioning (`docs/source/development.rst`); see "Public API".
- A changelog is kept in `CHANGES.rst`.

## Technology

Use the following languages, frameworks and libraries (all declared in `pyproject.toml`):

- Python `>=3.11` (`requires-python`). `.python-version` pins `3.13` for local development;
  `CI.yml` uses 3.13 as well. `pytest-qt` (group `test-qt`) is only installed for Python
  `<=3.13`, `ray` (group `docs`) only for `<=3.13` (`<=3.12` on Windows).
- Scientific Python stack: numpy, pandas, scipy, scikit-learn, scikit-image, matplotlib,
  networkx, shapely, lmfit, boost-histogram, fast-histogram.
- protobuf (metadata), pydantic-settings (configuration), lazy_loader (public API),
  qtpy (Qt abstraction), asdf, tifffile, ruamel.yaml, tomli-w, tqdm.
- Optional extras: `color`, `hdf5`, `http`, `napari`, `register` (open3d), `scatter_density`,
  `track` (trackpy), one Qt binding (`pyqt5`, `pyqt6`, `pyside2`, `pyside6`), `test`, `all`.
- Build: setuptools with setuptools_scm (version from git tags).
- Docs: Sphinx (furo theme, autosummary, napoleon, myst-nb), hosted on Read the Docs.
- CI: GitHub Actions (`.github/workflows/`, overview in `.github/workflows/README.md`).

## Tooling — always use these, never alternatives

- **uv** manages Python, the venv and dependencies locally. Never call `pip`, `python -m venv`,
  `conda` or bare `python`; prefix everything with `uv run`. `uv.lock` is local and gitignored:
  change dependencies only in `pyproject.toml` via `uv add` / `uv add --dev` / `uv remove`
  (`--group <name>` for the groups `test`, `test-qt`, `lint`, `typing`, `benchmark`, `docs`;
  `dev` combines pre-commit, `test`, `lint` and `typing`). CI and the published install paths
  deliberately use pip and conda (`environment.yml`, `environment_dev.yml`); do not change that.
- **black** is the only formatter. Never run `ruff format`.
- **ruff** is the only linter, including import sorting (no isort, flake8, pylint).
  Black and ruff share `line-length = 88` in `pyproject.toml`; keep them in sync.
- **pyrefly** is the only type checker (`preset = "strict"`).
- **pytest** is the test runner; **coverage** measures coverage (`fail_under = 90`, same in CI).
- **pre-commit** runs whitespace/yaml/toml/large-file hooks, black (incl. notebooks), ruff and
  pyrefly (`.pre-commit-config.yaml`).

## Architecture

The project is a Python library (`src/locan`, imported as `locan`) with a command-line
interface (`locan` console script → `locan.__main__:main`); all specifications are in
`pyproject.toml`.

- New code should follow the existing folder structure.

### Repository structure

Single-package repository with a `src` layout (no uv workspace). Key paths:

```
src/locan/                  # import as `locan`
├── __init__.py             # lazy_loader entry point (see "Public API")
├── __init__.pyi            # the public API stub: every exported name, one explicit re-export each
├── __main__.py             # CLI: draw_roi_mpl, rois, check, napari, show_versions, test
├── configuration.py        # pydantic-settings `Configuration`, exposed as `locan.config`
├── constants.py            # PropertyKey/PROPERTY_KEYS, file-format key maps, enums (FileType, HullType, RenderEngine)
├── dependencies.py         # optional-dependency machinery: HAS_DEPENDENCY, needs_package, QtBindings
├── datasets.py             # example datasets (download via httpx)
├── locan_types.py          # type definitions
├── analysis/               # analysis classes, all subclasses of `_Analysis` (analysis_base.py)
├── data/                   # LocData, Image, metadata (*.proto, *_pb2.py), regions/, hulls/, adapter/
├── locan_io/               # file IO: files.py (Files), locdata/<format>_io.py, load_locdata, manifest
├── process/                # aggregate (histogram), bins (Bins), filter, register, tracking, cluster/, properties/, transform/
├── rois/                   # Roi (region of interest tied to a LocData object)
├── simulation/             # simulate_locdata.py, simulate_drift.py
├── visualize/              # render.py dispatches on RenderEngine → render_mpl/, render_napari/; colormaps
├── gui/                    # Qt file dialogs
├── utils/                  # format, miscellaneous, rotation, statistics, system_information
├── scripts/                # CLI subcommands wired in __main__.py
├── licences/               # third-party licence texts
└── tests/                  # minimal in-package smoke suite + `locan.tests.test()` runner
tests/                      # extended suite, mirrors the subpackages; fixtures in tests/conftest.py
tests/test_data/            # static test data: sample files per format, asdf datasets, legacy metadata
docs/                       # Sphinx: conf.py, source/*.rst, tutorials/notebooks/*.ipynb
benchmarks/                 # asv benchmarks (benchmarks/README.md)
docker/                     # Dockerfiles for testing and deployment (docker/README.md)
tools/                      # maintainer scripts, e.g. script_make_init.py
.github/                    # issue and PR templates, workflows for GitHub Actions
.claude/                    # Claude Code settings, format hook, skill `add-analysis`
pyproject.toml              # packaging, dependencies and all tool configuration
CHANGES.rst                 # changelog
README.md                   # project introduction
CONTRIBUTING.md             # contribution details
```

Large modules: `data/regions/region.py` (~3500 lines), `simulation/simulate_locdata.py`
(~1800), `data/locdata.py` (~1400), `analysis/drift.py`, `process/bins.py` (~1100 each).
Grep first and read by line range; do not load whole.

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
  for module attributes via a module-level `__getattr__` (see `visualize/colormap.py`).
  Importing `locan` must emit no `DeprecationWarning` from `locan.*` (tested in
  `tests/test_locan_init.py`).

### Analysis classes (`src/locan/analysis/`)

Every analysis is a subclass of `_Analysis` (`analysis_base.py`). Follow an existing one such as
`localizations_per_frame.py`; the full checklist is `.claude/skills/add-analysis/SKILL.md`.

- `__init__(self, meta=None, <params>)` → `parameters = self._get_parameters(locals())` →
  `super().__init__(**parameters)`; `self.results = None`.
- A private module-level helper `_<name>(...)` does the computation.
- `compute(self, locdata) -> Self` sets `self.results` (a `_Results` dataclass or DataFrame) and
  returns `self`. Empty `locdata` is handled: log a warning, leave `results` None.
- Optional: `plot(ax=None, **kwargs) -> Axes`, `hist(...)`, `fit_distributions(...)`, `report()`.
- Metadata is `AMetadata` (protobuf `metadata_analysis.proto`); `_Analysis` records the
  parameters in it.

### Optional dependencies

- Declared as extras in `pyproject.toml`; detected once in `dependencies.py` (`HAS_DEPENDENCY`).
- Module pattern: `if HAS_DEPENDENCY["napari"]: import napari` at the top, and
  `@needs_package("napari")` on every public function that needs it (raises `ImportError`).
- Tests: `pytest.importorskip("napari")` at module level or
  `@pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="requires open3d")`.
- The Qt binding is chosen via `config.qt_binding` / the `QT_API` environment variable and
  used through qtpy; no module in `src/locan` imports a Qt binding directly.

### Data model

- Localization properties are named by `constants.PropertyKey` (`position_x`, `frame`,
  `intensity`, ...), each with a `PropertyDescription`.
- Property names use reverse notation: coordinate or statistics identifiers are appended
  (`position_x_mean_mean`, see `docs/source/properties.rst`).
- Metadata is protobuf. Sources are `data/metadata.proto`, `analysis/metadata_analysis.proto`,
  `locan_io/locdata/manifest.proto`. The generated `*_pb2.py` / `*_pb2.pyi` are checked in and
  excluded from ruff, pyrefly and coverage. A `.proto` change requires regenerating them.
  Unclear: the protoc command is not documented in the repository (`CHANGES.rst` records the
  last bump as protoc 34.1); ask before changing a `.proto` file.
- `Region` class hierarchy in `data/regions/region.py`; `Roi` (`rois/roi.py`) ties a region to
  a `LocData` object.
- Configuration is read at import from `[tool.locan]` in `pyproject.toml`, a
  `locan_config.toml`, a `.env` file, `locan_*` environment variables, or defaults
  (`configuration.py`).
- `src/locan/_version.py` is written by setuptools_scm and gitignored.

## Links with information

| Topic | Document |
| ----- | -------- |
| Complete documentation | `https://locan.readthedocs.io` |
| Development rules (kept consistent with this file) | `docs/source/development.rst` |
| Installation | `docs/source/installation.rst` |
| Design, data structures, properties, metadata | `docs/source/design.rst`, `datastructures.rst`, `properties.rst`, `metadata.rst` |
| Command-line interface | `docs/source/command_line_interface.rst` |
| Contribution details | `CONTRIBUTING.md` |
| CI workflows | `.github/workflows/README.md` |
| Docker | `docker/README.md` |
| Benchmarks (asv) | `benchmarks/README.md` |
| Checklist for a new analysis class | `.claude/skills/add-analysis/SKILL.md` |

## Commands

The commands below are written for a POSIX shell. Windows PowerShell has no `&&`: chain with
`; if ($?) { ... }` or use Git Bash.

| Command | Purpose |
| ------- | ------- |
| `uv sync --extra all --extra pyside6` | Install the full dev environment: `dev` group, all optional dependencies, one Qt binding |
| `uv run pre-commit install` | Install the git hooks (once) |
| `uv run python tools/script_make_init.py` | Run a Python script (this one prints the re-export blocks for `__init__.pyi`) |
| `uv run locan -h` | Command-line interface help |
| `uv run pytest tests/process/test_bins.py::test_from_bin_edges -xvs` | Run a single test — prefer while iterating |
| `uv run pytest tests/process/test_bins.py -xvs` | Run a test file |
| `uv run pytest` | Run all tests (both suites; `gui`, `visual`, `network` deselected) |
| `uv run pytest -x --lf` | Stop on first failure, rerun last failures |
| `uv run pytest -m "not slow and not gui and not visual and not network"` | Skip slow tests as well (`-m` replaces the default selection) |
| `uv run pytest -m network tests/test_datasets.py` | Dataset download tests (need network) |
| `uv run coverage run -m pytest && uv run coverage report` | Local test coverage (fails below 90 %) |
| `uv run ruff check --fix .` | Lint (incl. import sorting) |
| `uv run black .` | Format (also code cells in notebooks: `black[jupyter]`) |
| `uv run pyrefly check` | Typecheck |
| `uv run ruff check --fix . && uv run black . && uv run pyrefly check && uv run pytest` | Run all checks |
| `uv run pre-commit run --all-files` | Run all checks through pre-commit |
| `uv sync --group docs --extra all --extra pyside6 && uv run sphinx-build -b html docs docs/_build/html` | Create documentation (slow: executes all notebooks) |

## Boundaries

### Never

- Commit secrets, credentials or tokens.
- Edit generated files (`*_pb2.py`, `*_pb2.pyi`, `src/locan/_version.py`).
- Use destructive git operations unless explicitly requested.
- Commit or push unless asked.
- Publish packages (`uv publish`, `twine`); PyPI uploads go through `deploy_to_pypi.yml`.

### Do Not Modify

- Do not touch: `*_pb2.py*`, `src/locan/_version.py`, `uv.lock`.
- Do not touch `.github/workflows/*` — except when the task is about CI.
- Do not execute or re-save `docs/tutorials/notebooks/*.ipynb` unless asked (`black .`
  formatting their code cells is intended). If changed, save with cleared output.
- Do not reformat `tests/test_data/*` (excluded from the whitespace hooks).

### Requires Explicit Approval

- Large refactors across subpackages.
- New runtime dependencies; dev/test tools go in dependency groups.
- Renaming or removing exported names, or changing their signature incompatibly.
- Changing a `.proto` file (see "Data model").
- Changing test configuration (`[tool.pytest.ini_options]`, `[tool.coverage.*]`, markers).

## Guidelines

Work as a pair programmer, not a silent code generator:

- **Think from first principles**: Don't settle for the first solution; question assumptions
  about the true nature of the problem.
- **Requirements first**: Verify requirements before implementing, especially when writing or
  changing tests.
- **Research before guessing**: Search the codebase for similar patterns; check online sources
  for verification.
- **Discuss before deciding**: When strategy is unclear, present options and trade-offs to the
  user instead of choosing silently.
- **Step-by-step for large changes**: Break down significant refactorings and get confirmation
  along the way.
- **Challenge assumptions**: If the user states something untrue, correct them directly.
- Read the relevant module and its tests before changing anything.
- Make sure a code change is minimal — only what the task requires. Don't reformat or refactor
  unrelated code.
- If you find a bug or improvement opportunity outside the task scope:
  1. Note it in a comment
  2. Do not fix it
  3. Mention it in your response for human follow-up

### What to Avoid

- **Hard-wrapping prose on GitHub** — in issues, pull requests and comments, GitHub renders
  every line break, so keep each paragraph on a single line.
- **Overwriting `.env` or `locan_config.toml` files** without explicit user confirmation
  (locan reads its configuration from them).
- **Creating new files** when editing existing ones would suffice.
- **Global mutable state** in library code. The existing module-level state is `locan.config`
  and the counter `locan.locdata_id`; do not add more.
- **Unnecessary dependencies** — check if existing code suffices first.
- **Tests coupled to implementation details** — the PR checklist asks for a regression test
  for bug fixes; write it against observable behavior, not internals: no private attributes,
  no patched machinery, no fake objects mirroring the code under test.
  If you catch yourself building scaffolding to observe an internal mechanism, stop — find the
  user-visible effect to assert on, or skip the test and give the reason in the pull request.
  Before writing a test, read a recent one in the same file and copy its shape.

### Coding environment

- Use uv to set up the Python environment: `uv sync --extra all --extra pyside6`.
- Use `uv run <cmd>` for all commands — never activate the venv manually.
- Add dependencies with `uv add <pkg>`, not pip install.
- `uv.lock` is generated by `uv lock` / `uv sync` and is gitignored: never commit it.
- `.claude/settings.json` sets `MPLBACKEND=Agg` and `QT_QPA_PLATFORM=offscreen` for Claude Code
  sessions and runs `.claude/hooks/format_file.py` (ruff + black on the edited `.py` file)
  after every edit. Other agents have to format themselves (see "Coding instructions").

### Coding instructions

- Keep code comments in English.
- Include full type hints on all functions, methods and class attributes, including `-> None`.
- Use modern syntax: `list[int]`, `X | None`,
  `from collections.abc import Iterable, Sequence, Mapping`.
- Use f-strings, not `.format()` or `%`. Use `pathlib.Path` for paths.
- No `assert` for runtime checks in library code (ruff `S101`); raise specific exceptions with
  clear messages. The existing exception is type narrowing:
  `assert x is not None  # type narrowing # noqa: S101`.
- Imports at top of file (ruff `E402`), including the conditional imports of optional
  dependencies. Import aliases are enforced by ruff: `np`, `pd`, `plt`, `mpl`, `nx`, `sp`,
  `bh`, `o3d`.
- Guard heavy type-only imports with `TYPE_CHECKING`.
- Prefer action verbs for new functions and methods: `get_`, `load_`, `make_`, `simulate_`,
  `render_`, `transform_`, `compute`; predicates use `is_` / `has_`. The existing API is mixed
  (noun names such as `histogram` or `inertia_moments`, analysis helpers `_<name>`): do not
  rename existing names.
- Use `n_something` for "number of something".
- Code lines ≤ 88 chars (black); docstring lines ≤ 75 chars (`development.rst`; not enforced
  by ruff).
- No mutable default arguments.
- Comment only when context is not readily apparent from the code. Generic purpose
  explanations do not qualify: `# Log for debugging` above `logging.debug(...)`,
  `# Validate for safety`, `# Retry for reliability` and similar provide no useful information.
  Useful comments record a specific constraint, invariant, compatibility quirk, or tradeoff.
  Before adding a comment, identify the misunderstanding or incorrect change it would prevent.
  Do not narrate code, repeat code, or invent a rationale in comments. Keep necessary
  explanations short, precise and at the source of truth; do not repeat them at each call site.
- Always format and check Python files immediately after writing or editing them:
  `uv run ruff check --fix <file_path>` and `uv run black <file_path>`. Do this for every Python
  file you create or modify, before moving on to the next step.

### Type Checking

- Use pyrefly in strict mode; do not use mypy.
- pyrefly strict must stay clean. New or edited lines suppress errors only with a scoped code
  and reason: `# pyrefly: ignore[bad-assignment]  # reason: upstream stub is wrong`.
- The code base still contains hundreds of bare `# type: ignore` comments from the mypy era.
  They are legacy: leave them alone unless you are already changing that line.
- Avoid `Any` in new code; narrow it immediately and explain in a comment if unavoidable.
  (Existing code uses `Any` widely, e.g. for `**kwargs`.)
- `from __future__ import annotations` in every source module (present in all but a few
  near-empty ones).
- Never weaken `[tool.pyrefly]` or `[tool.ruff]` in `pyproject.toml` to make errors disappear.
  Same for `# noqa`: always with a rule code, e.g. `# noqa: S603`.
- In `tests/`, pyrefly does not check unannotated functions; test functions and fixtures are
  written without type hints.

### Testing instructions

- Use `pytest` for tests; do not use `unittest`.
- Two suites, both in `testpaths`: `src/locan/tests` is a minimal smoke suite shipped with the
  package (`locan test` CLI); `tests/` is the extended suite and mirrors the subpackages.
  New tests go in `tests/<subpackage>/test_<module>.py`.
- Naming: `test_<module>.py` / `test_<function>_<scenario>()`.
- Fixtures: shared ones are session-scoped in `tests/conftest.py` (`locdata_2d`, `locdata_3d`,
  `locdata_blobs_2d`, `locdata_dSTORM_2d`, `locdata_two_cluster_2d`, ...): never mutate them,
  use `deepcopy(locdata)`. Fixtures used by one module only are defined in that test file.
- Test data lives in `tests/test_data`; reach it via `from tests import TEST_DIR`.
- Markers (`--strict-markers`): `gui`, `visual` and `network` are deselected by default
  (`addopts`); `slow` is opt-out. Mark anything that opens a window, needs a human or downloads
  data. `pytest-timeout` kills any test running longer than 600 s.
- Add or update tests for the code you change, even if nobody asked.
- Target exactly 100 % coverage of what the PR changes — no more, no less (the repository-wide
  gate is 90 %). Every changed or added behaviour must have a test; every test must fail
  without the PR's change.
- Do not add tests for pre-existing logic that was already present before the PR.
- Run the relevant tests before suggesting a final change.
- Do not test standard-library or third-party functions. The exception is deliberate behaviour
  or integration tests, which may cross those boundaries by design.
- Use `@pytest.mark.parametrize` for multiple similar inputs — consolidate tests that only
  differ in input/expected values into a single parametrized test.
- Tests are deterministic and offline: seed numpy (`rng = np.random.default_rng(seed)`), no
  network unless marked `network`.
- Test functions need no docstring (ruff `D` rules are not enabled); fixtures get one.
- Don't delete or loosen an existing test to make it pass. If a test seems wrong, say why and
  ask.

### Docs and changelog

- NumPy-style docstrings on every module, class and public function with `Parameters`,
  `Returns`, `Raises`, `Attributes`; a short `Examples` block on public functions where useful
  (doctest format). Not enforced by ruff (`D` is not selected); Sphinx renders them via napoleon.
- API docs are generated by autosummary: the `__init__.py` docstring of each subpackage lists
  its modules in an `.. autosummary::` block; keep the lists current when adding modules.
- Tutorials are Jupyter notebooks in `docs/tutorials/notebooks/`, executed by the docs build
  (`nb_execution_mode = "force"`).
- Keep the changelog in reStructuredText in `CHANGES.rst`: add a one-line entry under the top
  `<version> - dev` section in the fitting subsection: `New Features`, `API Changes`,
  `Bug Fixes` (prefixed `fix: ...`), or `Other Changes and Additions`.
- Releases: the version comes from git tags (setuptools_scm); `deploy_to_pypi.yml` builds and
  uploads to PyPI when a GitHub release is published. Unclear: the remaining release steps
  are not documented in the repository; ask before preparing a release.

### Commit

- Never commit unless asked.
- Commit messages follow Conventional Commits with an imperative subject. The history uses
  `type: description` without a scope (`fix: ...`, `feat: ...`, `refactor: ...`); the body
  explains why.

### Creating Pull Requests

Always use the repository's PR template
([`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md)) with its **Summary**
and **Checklist** sections. Do not invent alternative formats.

## Anti-Patterns

### Error Handling

- Never catch and swallow exceptions — log and re-raise, or raise a specific exception with a
  clear message.
- Never use bare `except:` — always specify the exception type (ruff `E722`, `BLE`).
- Never use `print()` for debug output — use the module logger:
  `logger = logging.getLogger(__name__)`, `logger.debug()`. `print` is reserved for output the
  user asked for: `report()` / `print_*` methods, `show_versions` and the CLI.

## Before Claiming a Task Complete

Run the following in this order, make all pass and review your own diff:

```bash
uv run ruff check --fix .
uv run black .
uv run pyrefly check
uv run pytest
```

Then add an entry to `CHANGES.rst` under the current dev section (see "Docs and changelog").
If something fails that is unrelated to your change, say so explicitly instead of fixing it
silently.

## Security Requirements

These are non-negotiable. If you encounter a conflict with other requirements,
security requirements take precedence.

### Secrets and Credentials

- The repository contains no secrets. CI reads `secrets.CODECOV_TOKEN`; the PyPI upload
  authenticates through the GitHub environment `pypi` (`id-token: write`), not a stored token.
- Never put tokens or credentials into code, tests, notebooks, workflow files or configuration
  files. `.env` and `.claude/settings.local.json` are gitignored; keep them out of commits.
- ruff's flake8-bandit rules (`S`) apply to library code (`subprocess`, `eval`, ...). Do not
  silence them wholesale; suppress a single line only with a rule code and a reason.
- Library code accesses the network only in `datasets.py` (httpx download of the example
  datasets). Do not add other network access without asking.

### Reporting

If you notice a potential security issue while working on something else:
mention it immediately in your response, do not attempt to fix it silently.

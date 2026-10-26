# Contributing to Locan

Thank you for considering a contribution.

- **Bugs and feature requests**: open a [GitHub issue](https://github.com/super-resolution/Locan/issues).
  Questions go to [GitHub discussions](https://github.com/super-resolution/Locan/discussions).
- **Development rules** (environment, code checks, testing, conventions) are given in
  `docs/source/development.rst` and rendered at
  [documentation on development](https://locan.readthedocs.io/en/latest/source/development.html).
- **AI coding agents** (and humans who like checklists) follow [AGENTS.md](AGENTS.md); it is the
  authoritative, repository-specific summary of layout, patterns and the definition of done.

Quick start with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra all --extra pyside6
uv run pre-commit install
uv run ruff check --fix . && uv run black . && uv run pyrefly check && uv run pytest
```

Pull requests should pass the checklist in the PR template and add a line to `CHANGES.rst`.

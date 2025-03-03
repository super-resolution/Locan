import subprocess
from textwrap import dedent

import pytest

# ruff: noqa: W293

pytestmark = pytest.mark.skip(
    "Explicit test of subpackage imports. Implicitly tested by other tests."
)


subpackages = [
    "analysis",
    "configuration",
    "constants",
    "data",
    "datasets",
    "dependencies",
    "gui",
    "locan_io",
    "locan_types",
    "process",
    "rois",
    "scripts",
    "simulation",
    "tests",
    "utils",
    "visualize",
]


def _temp_script_file(subpackage):
    script = f"""\
    from importlib import import_module
    def _import_subpackage(subpackage):
        module = import_module(name=subpackage, package="locan")
        assert module
        print(module, dir(module))
    _import_subpackage(subpackage="{subpackage}")
    """
    return dedent(script)


def test_system_import(tmpdir):
    file = tmpdir.mkdir("sub").join("script.py")
    for subpackage_string in subpackages:
        file.write(_temp_script_file(subpackage=subpackage_string))
        result = subprocess.run(  # noqa: S602
            ["python", file], shell=True, capture_output=True, text=True
        )
        result.check_returncode()


# individual imports:
def test_system_import_data(tmpdir):
    file = tmpdir.mkdir("sub").join("script.py")
    file.write(_temp_script_file(subpackage="data"))
    result = subprocess.run(  # noqa: S602
        ["python", file], shell=True, capture_output=True, text=True
    )
    try:
        result.check_returncode()
    except subprocess.CalledProcessError:
        print(result.stderr)


def test_system_import_gui(tmpdir):
    file = tmpdir.mkdir("sub").join("script.py")
    file.write(_temp_script_file(subpackage="gui"))
    result = subprocess.run(  # noqa: S602
        ["python", file], shell=True, capture_output=True, text=True
    )
    try:
        result.check_returncode()
    except subprocess.CalledProcessError:
        print(result.stderr)

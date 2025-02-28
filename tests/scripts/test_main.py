import subprocess

import pytest

pytestmark = pytest.mark.slow


def test_entrypoint_from_sys():
    exit_status = subprocess.run(  # noqa S603
        "locan", capture_output=True, encoding="utf-8"
    )
    assert exit_status.stdout[:47] == "This is the command line entry point for locan."
    exit_status.check_returncode()


def test_main():
    exit_status = subprocess.run(  # noqa S603
        ["locan", "--help"], capture_output=True, encoding="utf-8"
    )
    assert exit_status.stdout.startswith("usage:")
    exit_status.check_returncode()

    exit_status = subprocess.run(  # noqa S603
        ["locan", "--version"], capture_output=True, encoding="utf-8"
    )
    assert exit_status.stdout.startswith("locan version")
    exit_status.check_returncode()

    exit_status = subprocess.run(  # noqa S603
        ["locan", "--info"], capture_output=True, encoding="utf-8"
    )
    for item in ["Locan:", "Python:", "System:", "Python dependencies:"]:
        assert item in exit_status.stdout
    exit_status.check_returncode()

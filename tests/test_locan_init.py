import subprocess
import warnings

import pytest

import locan
from locan.__main__ import main


def test_version():
    try:
        assert isinstance(locan._version.version, str)
        assert isinstance(locan._version.version_tuple, tuple)
    except AttributeError:
        warnings.warn(
            "AttributeError was raised; "
            "most likely because locan._version is not available.",
            stacklevel=2,
        )
    assert locan.__version__
    intersection = set(dir(locan)).difference(set(locan.__all__))

    # print(locan.__version__)
    # print(len(dir(locan)))  # currently: 211
    # print(dir(locan))
    # print(len(locan.__all__))
    # print(locan.__all__)
    # print(intersection)

    assert not intersection


def test_entrypoint(capfd):
    main([])
    captured = capfd.readouterr()
    assert captured.out[:47] == "This is the command line entry point for locan."


@pytest.mark.slow
def test_entrypoint_from_sys():
    exit_status = subprocess.run(  # noqa S603
        "locan", capture_output=True, encoding="utf-8"
    )
    assert exit_status.stdout[:47] == "This is the command line entry point for locan."
    exit_status.check_returncode()

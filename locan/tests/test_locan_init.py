import os

import locan
from locan.__main__ import main


def test_version():
    try:
        assert isinstance(locan._version.version, str)
        assert isinstance(locan._version.version_tuple, tuple)
    except AttributeError:
        pass
    assert locan.__version__
    assert locan.__all__
    # print(dir(locan))
    # print(locan.__version__)
    # print(locan.__all__)


def test_entrypoint(capfd):
    main([])
    captured = capfd.readouterr()
    assert captured.out[:47] == "This is the command line entry point for locan."


def test_entrypoint_from_sys(capfd):
    exit_status = os.system("locan")
    captured = capfd.readouterr()
    assert captured.out[:47] == "This is the command line entry point for locan."
    assert exit_status == 0

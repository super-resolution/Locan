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
    # print(locan.__version__)
    # print(len(dir(locan)))
    # print(dir(locan))
    # print(len(locan.__all__))
    # print(locan.__all__)

    assert all(item in locan.__dict__ for item in locan.__all__)
    assert all(item in locan.__dict__ for item in locan.submodules)
    # to double check:
    # stoplist = ['annotations', 'logging', 'import_module', 'Path', '_version',
    # 'locdata_id', 'submodules', 'submodule', 'locan_types', 'module_', 'scripts']
    # all([
    #     item in stoplist
    #     for item in locan.__dict__
    #     if (item not in locan.__all__
    #         and item not in locan.submodules
    #         and not item.startswith("__")
    #         )
    # ])


def test_entrypoint(capfd):
    main([])
    captured = capfd.readouterr()
    assert captured.out[:47] == "This is the command line entry point for locan."


def test_entrypoint_from_sys(capfd):
    exit_status = os.system("locan")
    captured = capfd.readouterr()
    assert captured.out[:47] == "This is the command line entry point for locan."
    assert exit_status == 0

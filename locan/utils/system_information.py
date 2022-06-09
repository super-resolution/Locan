"""
Utility methods to print system and dependency information.

adapted from :func:`pandas.show_versions` (`locan/licences/PANDAS.rst`)
(BSD 3-Clause License, Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development
Team, Copyright (c) 2011-2021, Open source contributors.)
and from :func:`scikit-learn.show_versions` (`locan/licences/SCIKIT-LEARN.rst`)
(BSD 3-Clause License, Copyright (c) 2007-2020 The scikit-learn developers.)
"""
import importlib
import locale
import os
import platform
import struct
import sys

from locan import __version__ as locan_version
from locan.dependencies import EXTRAS_REQUIRE, INSTALL_REQUIRES

__all__ = ["system_info", "dependency_info", "show_versions"]


def system_info(verbose=True):
    """
    Return system information.

    Parameters
    ----------
    verbose : bool
        If True information on node and executable path are added.

    Return
    ------
    dict
        System information
    """
    uname_result = platform.uname()
    language_code, encoding = locale.getlocale()

    sys_info = {
        "python-bits": struct.calcsize("P") * 8,
        "system": uname_result.system,
        "release": uname_result.release,
        "version": uname_result.version,
        "machine": uname_result.machine,
        "processor": uname_result.processor,
        "byteorder": sys.byteorder,
        "LC_ALL": os.environ.get("LC_ALL"),
        "LANG": os.environ.get("LANG"),
        "LOCALE": {"language-code": language_code, "encoding": encoding},
    }

    if verbose:
        sys_info.update(
            {"node": uname_result.node, "executable": sys.executable,}
        )

    return sys_info


def dependency_info(extra_dependencies=True, other_dependencies=None):
    """
    Overview of the installed version of main dependencies.

    Parameters
    ----------
    extra_dependencies : bool
        Include extra dependencies as specified in setup.py

    other_dependencies : list or None
        Include other module names.

    Returns
    -------
    dict
        Version information on relevant Python libraries
    """
    deps = INSTALL_REQUIRES.copy()

    if extra_dependencies:
        deps = deps.union(EXTRAS_REQUIRE)

    if other_dependencies:
        deps = deps.union(other_dependencies)

    deps_info = {}
    for modname in deps:
        try:
            deps_info[modname] = importlib.metadata.version(modname)
        except importlib.metadata.PackageNotFoundError:
            deps_info[modname] = None
    return deps_info


def show_versions(
    locan=True,
    python=True,
    system=True,
    dependencies=True,
    verbose=True,
    extra_dependencies=True,
    other_dependencies=None,
):
    """
    Print useful debugging information on system and dependency versions.

    Parameters
    ----------
    locan : bool
        Show locan version
    python : bool
        Show python version
    system : bool
        Show system information
    verbose : bool
        If True information on node and executable path are added.
    dependencies : bool
        Show main dependencies
    extra_dependencies : bool
        Include extra dependencies as specified in setup.py if True.
    other_dependencies : list or None
        Include other module names.

    Returns
    -------
    None
    """

    locan_info = {"version": locan_version}
    python_info = {"version": platform.python_version()}
    sys_info = system_info(verbose)
    deps_info = dependency_info(extra_dependencies, other_dependencies)

    if locan:
        print("\nLocan:")
        for key, value in locan_info.items():
            print(f"{key:>10}: {value}")

    if python:
        print("\nPython:")
        for key, value in python_info.items():
            print(f"{key:>10}: {value}")

    if system:
        print("\nSystem:")
        for key, value in sys_info.items():
            print(f"{key:>10}: {value}")

    if dependencies:
        print("\nPython dependencies:")
        for key, value in deps_info.items():
            print(f"{key:>10}: {value}")

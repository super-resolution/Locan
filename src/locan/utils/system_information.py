"""
Utility methods to print system and dependency information.

adapted from :func:`pandas.show_versions` (`locan/licences/PANDAS.rst`)
(BSD 3-Clause License, Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development
Team, Copyright (c) 2011-2021, Open source contributors.)
and from :func:`scikit-learn.show_versions` (`locan/licences/SCIKIT-LEARN.rst`)
(BSD 3-Clause License, Copyright (c) 2007-2020 The scikit-learn developers.)
"""
from __future__ import annotations

import importlib
import locale
import os
import platform
import struct
import sys
from typing import Any

from locan import __version__ as locan_version
from locan.dependencies import EXTRAS_REQUIRE, INSTALL_REQUIRES

__all__: list[str] = ["system_info", "dependency_info", "show_versions"]


def system_info(verbose: bool = True) -> dict[str, Any]:
    """
    Return system information.

    Parameters
    ----------
    verbose
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
        sys_info.update({"node": uname_result.node, "executable": sys.executable})

    return sys_info


def dependency_info(
    extra_dependencies: bool = True, other_dependencies: list[str] | None = None
) -> dict[str, Any]:
    """
    Overview of the installed version of main dependencies.

    Parameters
    ----------
    extra_dependencies
        Include extra dependencies as specified in setup.py

    other_dependencies
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

    deps_info: dict[str, str | None] = {}
    for modname in deps:
        try:
            deps_info[modname] = importlib.metadata.version(modname)
        except importlib.metadata.PackageNotFoundError:
            deps_info[modname] = None
    return deps_info


def show_versions(
    locan: bool = True,
    python: bool = True,
    system: bool = True,
    dependencies: bool = True,
    verbose: bool = True,
    extra_dependencies: bool = True,
    other_dependencies: list[str] | None = None,
) -> None:
    """
    Print useful debugging information on system and dependency versions.

    Parameters
    ----------
    locan
        Show locan version
    python
        Show python version
    system
        Show system information
    verbose
        If True information on node and executable path are added.
    dependencies
        Show main dependencies
    extra_dependencies
        Include extra dependencies as specified in setup.py if True.
    other_dependencies
        Include other module names.
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

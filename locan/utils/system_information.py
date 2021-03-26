"""
Utility methods to print system and dependency information.

adapted from :func:`pandas.show_versions`
and from :func:`scikit-learn.show_versions`
"""
# License: BSD 3 clause

import platform
import sys
import importlib
import struct
import os
import locale

from locan.constants import INSTALL_REQUIRES, EXTRAS_REQUIRE
from locan import __version__ as locan_version


__all__ = ['system_info', 'dependency_info', 'show_versions']


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
        sys_info.update({
            "node": uname_result.node,
            "executable": sys.executable,
        })

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
    deps = INSTALL_REQUIRES

    if extra_dependencies:
        extra_deps = [value for key, value in EXTRAS_REQUIRE.items()]
        for ed in extra_deps:
            deps.extend(ed)

    if other_dependencies:
        for od in other_dependencies:
            deps.extend(od)

    deps_info = {}
    for modname in deps:
        try:
            if modname in sys.modules:
                module = sys.modules[modname]
            else:
                module = importlib.import_module(modname)
        except ImportError:
            module = None

        deps_info[modname] = getattr(module, "__version__", None)

    return deps_info


def show_versions(locan=True, python=True, system=True, dependencies=True,
                  verbose=True, extra_dependencies=True, other_dependencies=None):
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

    locan_info = {'version': locan_version}
    python_info = {'version': platform.python_version()}
    sys_info = system_info(verbose)
    deps_info = dependency_info(extra_dependencies, other_dependencies)

    if locan:
        print('\nLocan:')
        for key, value in locan_info.items():
            print(f"{key:>10}: {value}")

    if python:
        print('\nPython:')
        for key, value in python_info.items():
            print(f"{key:>10}: {value}")

    if system:
        print('\nSystem:')
        for k, stat in sys_info.items():
            print("{k:>10}: {stat}".format(k=k, stat=stat))

    if dependencies:
        print('\nPython dependencies:')
        for k, stat in deps_info.items():
            print("{k:>10}: {stat}".format(k=k, stat=stat))

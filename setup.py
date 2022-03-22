"""
setup.py used for "pip install -e ."
since PEP 517 doesn’t support editable installs.
"""
from setuptools import setup
setup(
    use_scm_version={"write_to": "locan/_version.py",
                     "fallback_version": "0.11-fallback"}
    )

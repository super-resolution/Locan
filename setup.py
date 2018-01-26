import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Surepy",
    version = "0.1dev",

    packages=find_packages(),
    scripts=[],

    package_data={
        # Include *.txt files in the test/test_data directory:
        'Surepy/tests/test_data': ['*.txt']
        },

    author = "Surepy Contributors",
    author_email = "",

    description = ("Single-molecule localization software under development"),
    long_description=read('README.rst'),

    license="",
    keywords = "fluorescence super-resolution single-molecule localization microscopy smlm storm dstorm palm paint",
    url = "",

    # integrating pytest
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
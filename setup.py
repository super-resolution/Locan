import os
from setuptools import setup, find_packages

def read(fname):
    '''
    Utility function to read the README file.
    '''
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

    author = "Surepy Developers",
    author_email = "",

    description = ("Single-molecule localization software under development"),
    long_description=read('README.rst'),

    license="BSD-3",
    keywords = "fluorescence super-resolution single-molecule localization microscopy smlm storm dstorm palm paint",
    url = "",

    # dependencies
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
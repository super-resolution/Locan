import os
from setuptools import setup, find_packages

#
# utility functions
#

def read(fname):
    '''
    Utility function to read the README file.
    '''
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

#
# Set constants for setup
#

# project name
NAME = "Surepy"

# version updated manually
VERSION = 0.1

# include packages
PACKAGES = find_packages()

# include files other than python code
PACKAGE_DATA = {
        # Include *.txt files in the test/test_data directory:
        'Surepy/tests/test_data': ['*.txt']
        }

#
SCRIPTS = []

# contact information
AUTHOR = "Surepy Developers"
AUTHOR_EMAIL = ""

# information about the project
DESCRIPTION = ("Single-molecule localization software under development")
LONG_DESCRIPTION = read('README.md')
LICENSE = "BSD-3"
KEYWORDS = "fluorescence super-resolution single-molecule localization microscopy smlm storm dstorm palm paint"
URL = ""

# dependencies
PYTHON_REQUIRES = '>=3.5'
SETUP_REQUIRES = ['pytest-runner']
INSTALL_REQUIRES = ['numpy>=1.8', 'pandas', 'matplotlib', 'protobuf']
TESTS_REQUIRE = ['pytest']

# entry points to register scripts
ENTRY_POINTS = '''
    [console_scripts]
    draw_roi=surepy.scripts.draw_roi:main
    '''



setup(
    name = NAME,
    version = VERSION,
    packages = PACKAGES,
    scripts = SCRIPTS,
    package_data = PACKAGE_DATA,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    license = LICENSE,
    keywords = KEYWORDS,
    url = URL,
    python_requires = PYTHON_REQUIRES,
    setup_requires = SETUP_REQUIRES,
    tests_require = TESTS_REQUIRE,
    install_requires = INSTALL_REQUIRES,
    entry_points = ENTRY_POINTS,
)
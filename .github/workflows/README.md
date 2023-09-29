# Workflows

## Continuous integration:

**1) CI**

    Continuous integration with formating, linting, type checking and testing.
    On linux with current python version
    Scheduled: on push, pull-request

**1) CI checks with black, ruff, mypy**

    Continuous integration with formating, linting and type checking.
    On linux with current python version
    Scheduled: on request

**1) CI with qt**

    Continuous integration with formating, linting and type checking.
    On linux with current python version
    Including all qt-related tests.
    Scheduled: on request


### Tests with Conda:

**1) Tests with ubuntu, Mambaforge**

    Create environment from environment.yml file with recommended python on ubuntu, 
    install locan from sources, and run tests.
    Scheduled: on push, pull-request

**2) Tests with python matrix, Mambaforge**

    Create environment from environment.yml file on OS- and python-matrix,
    install locan from sources, and run tests.
    Scheduled: once a week

**2) Tests with OS matrix, Mambaforge**

    Create environment from environment.yml file on OS- and python-matrix,
    install locan from sources, and run tests.
    Scheduled: once a month


### Tests with Pip:

**1) Tests with python matrix, pip**

    Install locan without extra dependencies on ubuntu and python-matrix, and run tests.
    Scheduled: once a week

**2) Tests with matrix, pip, extra dependencies**

    Install locan[all] on ubuntu and python-matrix, and run tests.
    Includes fixes for current dependency problems.
    Scheduled: on request

**3) Tests with matrix, pip, extra dependencies, qt**

    Install locan[all] on ubuntu and python-matrix, and run tests.
    Includes fixes for current dependency problems.
    Scheduled: once a week

**4) Tests with OS matrix, pip, extra dependencies, qt**

    Create environment from environment.yml file on OS- and python-matrix,
    install locan from sources, and run tests.
    Scheduled: once a month

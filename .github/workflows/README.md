# Workflows:

## CI

### Conda
**1) Test_conda**

    Create environment from environment.yml file with recommended python on ubuntu, 
    install locan from sources, and run tests.
    Scheduled: on push, pull-request

**2) Test_matrix_conda**

    Create environment from environment.yml file on OS- and python-matrix,
    install locan from sources, and run tests.
    Scheduled: once a week

### Pip

**1) Test_pip**
    
    Install locan without extra dependencies with recommended python on ubuntu, and run tests.
    Scheduled: on push, pull-request

**2) Test_matrix_pip**

    Install locan without extra dependencies on OS- and python-matrix, and run tests.
    Scheduled: once a week

**3) Test_matrix_pip_all**

    Install locan[all] on OS- and python-matrix, and run tests.
    Includes fixes for current dependency problems.
    Scheduled: once a week

**4) Test_matrix_pip_all_original**

    Install locan[all] on OS- and python-matrix, and run tests.
    Has on fixes for current dependency problems.
    Scheduled: workflow_dispatch
# Use an official image for a Python runtime that is based on debian
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bullseye

ARG PYTHON_VERSION
ARG ENVIRONMENT_NAME="locan"

ENV PYTHON_VERSION=${PYTHON_VERSION}

LABEL maintainer="LocanDevelopers" \
      project="locan" \
      python_version=${PYTHON_VERSION}

# set time zone to local time
RUN ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# Install gcc compiler and remove package manager cache
RUN apt-get update && \
    apt-get install gcc -yq && \
    # the following package is needed for slim-bullseye
    # when building wheels e.g. for scikit-image
    apt-get install -yq libgtk2.0-dev libgl1 && \
    # install git for setuptools_scm to deal with locan source distribution versioning
    apt-get install -yq git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . /locan

WORKDIR /locan

# Set up and activate virtual environment
ENV VIRTUAL_ENV "/opt/venv"
RUN python -m venv $VIRTUAL_ENV --upgrade-deps
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

# Update and install any needed packages and the project
RUN pip install setuptools pip-tools --trusted-host pypi.org --no-cache-dir && \
    pip install .[test]

# Volume for data
VOLUME ["/shared"]

# Run a command when the container launches
CMD today=$(date +"%Y-%m-%d") && \
    base="-linux-py"${PYTHON_VERSION}_$today && \
    pip-compile --extra test --output-file "/shared/requirements$base.txt" pyproject.toml && \
    date > "/shared/versions$base.txt" && \
    locan show_versions -e -v >> "/shared/versions$base.txt" && \
    date > "/shared/test_results$base.txt" && \
    locan test >> "/shared/test_results$base.txt"

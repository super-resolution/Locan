# Use an official image for a Python runtime that is based on debian
ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-slim-buster

ARG PYTHON_VERSION
ARG ENVIRONMENT_NAME="locan_python_py39"

ENV PYTHON_VERSION=${PYTHON_VERSION} \
    ENVIRONMENT_NAME=${ENVIRONMENT_NAME}

LABEL maintainer="LocanDevelopers" \
      project="locan" \
      python_version=${PYTHON_VERSION} \
      environment_name=${ENVIRONMENT_NAME}

# set time zone to local time
RUN ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# Install gcc compiler and remove package manager cache
RUN apt-get update && \
    apt-get install gcc -y && \
    apt-get install qt5-default -y && \
    # install git for setuptools_scm to deal with locan source distribution versioning
    apt-get install -qqy git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . /locan

WORKDIR /locan

# Set up and activate virtual environment
ENV VIRTUAL_ENV "/opt/venv"
RUN python -m venv $VIRTUAL_ENV
ENV PATH "$VIRTUAL_ENV/bin:$PATH"
	
# Update and install any needed packages and the project
RUN pip install pip setuptools --trusted-host pypi.org --upgrade --no-cache-dir && \
    pip install .[test]

# Volume for data
VOLUME ["/shared"]

# Run a command when the container launches
CMD today=$(date +"%Y-%m-%d") && \
    base=${ENVIRONMENT_NAME}_$today && \
    echo $PATH > "/shared/path_$base.txt" && \
    pip freeze --all > "/shared/requirements_$base.txt" && \
    date > "/shared/versions_$base.txt" && \
    locan show_versions -e -v >> "/shared/versions_$base.txt" && \
    date > "/shared/test_results_$base.txt" && \
    locan test >> "/shared/test_results_$base.txt"

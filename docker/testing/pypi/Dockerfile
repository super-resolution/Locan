# Use an official image for a Python runtime that is based on debian
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

ARG PYTHON_VERSION
# one of pyside2, pyqt5, pyside6, pyqt6
ARG QT_BINDING="pyside6"

ENV PYTHON_VERSION=${PYTHON_VERSION}

LABEL maintainer="LocanDevelopers" \
      project="locan" \
      python_version=${PYTHON_VERSION}

# set time zone to local time
RUN ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# Install gcc compiler and remove package manager cache
# Install libglib2.0-0 libegl1 for open source implementations of OpenGL and more
# install git for setuptools_scm to deal with locan source distribution versioning
RUN apt-get update && \
    apt-get install -yq \
        gcc \
        libglib2.0-0 \
        libegl1 \
        git && \
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
RUN pip install --no-cache-dir \
        setuptools \
        pip-tools && \
    pip install $QT_BINDING .[test]

# Volume for data
VOLUME ["/shared"]

# Run a command when the container launches
CMD today=$(date +"%Y-%m-%d") && \
    base="-linux-py"${PYTHON_VERSION}_$today && \
    requirements_file_name="/shared/requirements$base" && \
    pip-compile --extra test --output-file "$requirements_file_name.txt" pyproject.toml && \
    lock_file="$requirements_file-frozen.txt" && \
    pip freeze --requirement "$requirements_file_name.txt" > $lock_file && \
    date > "/shared/versions$base.txt" && \
    locan show_versions -e -v >> "/shared/versions$base.txt" && \
    date > "/shared/test_results$base.txt" && \
    locan test >> "/shared/test_results$base.txt" && \
    pytest -s >> "/shared/pytest_results$base.txt"

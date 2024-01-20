# Use an official image for micromamba that is based on debian
FROM mambaorg/micromamba:bookworm-slim AS base

ARG PYTHON_VERSION=3.11
ENV PYTHON_VERSION=${PYTHON_VERSION}

USER root

# Set time zone to local time
RUN ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# Install GL library and remove package manager cache
RUN apt-get update && \
    apt-get install libgl1-mesa-dev -y && \
    # install git for setuptools_scm to deal with locan source distribution versioning
    apt-get install -qqy git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER


FROM base AS build

# Copy the current directory contents into the container
COPY --chown=$MAMBA_USER:$MAMBA_USER . locan

RUN micromamba install  \
    --yes  \
    --name base  \
    --file ./locan/environment.yml  \
    python=${PYTHON_VERSION} && \
    micromamba clean --yes --all

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN pip install ./locan/.


FROM base

ARG MAMBA_DOCKERFILE_ACTIVATE=1

LABEL maintainer="LocanDevelopers" \
      project="locan" \
      python_version=${PYTHON_VERSION}

COPY --from=build /opt/conda /opt/conda

# Volume for data
VOLUME ["/shared"]

CMD today=$(date +"%Y-%m-%d") && \
    base="-linux-py"${PYTHON_VERSION}_$today && \
    micromamba env export --name base --explicit > "/shared/environment$base.lock" && \
    date > "/shared/versions$base.txt" && \
    locan show_versions -e -v >> "/shared/versions$base.txt" && \
    date > "/shared/test_results$base.txt" && \
    locan test >> "/shared/test_results$base.txt"

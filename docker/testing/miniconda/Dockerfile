# Use an official image for miniconda3 that is based on debian
FROM continuumio/miniconda3 AS base

ARG PYTHON_VERSION=3.12
ARG ENVIRONMENT_NAME="locan"

ENV PYTHON_VERSION=${PYTHON_VERSION} \
    ENVIRONMENT_NAME=${ENVIRONMENT_NAME}

# set time zone to local time
RUN ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# Install GL library and remove package manager cache
RUN apt-get update && \
    apt-get install libgl1-mesa-dev -y && \
    # install git for setuptools_scm to deal with locan source distribution versioning
    apt-get install -qqy git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


FROM base AS build

#Activate bash shell for correct use of conda commands
SHELL [ "/bin/bash", "--login", "-c" ]

# Copy the current directory contents into the container
COPY . locan

WORKDIR /locan

# Update conda & Create environment from environment.yml & Install the project
RUN conda init bash && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda update -n base -c conda-forge conda && \
    conda info && \
    conda create -n $ENVIRONMENT_NAME python=$PYTHON_VERSION -c conda-forge && \
    conda env update -n $ENVIRONMENT_NAME --file environment.yml && \
    conda activate $ENVIRONMENT_NAME && \
    pip install . && \
    conda clean -afy


FROM base

LABEL maintainer="LocanDevelopers" \
      project="locan" \
      python_version=${PYTHON_VERSION} \
      environment_name=${ENVIRONMENT_NAME}

COPY --from=build /opt/conda /opt/conda

SHELL [ "/bin/bash", "--login", "-c" ]

# Volume for data
VOLUME ["/shared"]

# Run a command when the container launches
CMD conda init bash && \
    conda activate $ENVIRONMENT_NAME && \
    today=$(date +"%Y-%m-%d") && \
    base="-linux-py"${PYTHON_VERSION}_$today && \
    conda env export > "/shared/environment$base.lock" && \
    date > "/shared/versions$base.txt" && \
    locan show_versions -e -v >> "/shared/versions$base.txt" && \
    date > "/shared/test_results$base.txt" && \
    locan test >> "/shared/test_results$base.txt"

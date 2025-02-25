# Use an official image for miniforge that is based on ubuntu
FROM condaforge/miniforge3:latest AS base

ARG PYTHON_VERSION=3.12

ENV PYTHON_VERSION=${PYTHON_VERSION} \
    USER="locan_user"

USER root

# Set time zone to local time
RUN ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# Install GL library and remove package manager cache
# install git for setuptools_scm to deal with locan source distribution versioning
RUN apt-get update && \
    apt-get install -yq \
        libgl1-mesa-dev \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash $USER

#Activate bash shell for correct use of conda commands
SHELL [ "/bin/bash", "--login", "-c" ]


FROM base AS build

# Copy the current directory contents into the container
COPY . locan

WORKDIR /locan

# Update conda & Create environment from environment.yml & Install the project
RUN conda init bash && \
    conda update -n base conda && \
    conda install python=$PYTHON_VERSION && \
    conda env update -n base --file environment.yml && \
    source activate base && \
    pip install . && \
    conda clean -afy


FROM base

LABEL maintainer="LocanDevelopers" \
      project="locan" \
      python_version=${PYTHON_VERSION}

COPY --from=build --chown=$USER:$USER /opt/conda /opt/conda

USER $USER

# Volume for data
VOLUME ["/shared"]

CMD today=$(date +"%Y-%m-%d") && \
    base="-linux-py"${PYTHON_VERSION}_$today && \
    conda env export > "/shared/environment$base.lock" && \
    date > "/shared/versions$base.txt" && \
    locan show_versions -e -v >> "/shared/versions$base.txt" && \
    date > "/shared/test_results$base.txt" && \
    locan test >> "/shared/test_results$base.txt"
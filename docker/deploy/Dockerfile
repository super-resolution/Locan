# Dockerfile for deploying jupyter lab, locan, napari and others
# in a conda environment as non-root user.
#
# This dockerfile is in parts adapted from dockerfiles by the Jupyter Development Team
# https://github.com/jupyter/docker-stacks
#
# Use an official image for miniforge that is based on ubuntu
FROM condaforge/miniforge3:latest AS base

ARG PYTHON_VERSION=3.12

ENV PYTHON_VERSION=$PYTHON_VERSION \
    USER="locan_user"
ENV HOME=/home/$USER

USER root

# Set time zone to local time
RUN ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

RUN apt-get update && \
    apt-get install -yq \
        # Install gcc compiler
        gcc \
        # Add necessary fonts for matplotlib/seaborn
        # See https://github.com/jupyter/docker-stacks/pull/380 for details
        fonts-liberation \
        # pandoc is used to convert notebooks to html files
        pandoc \
        # Install for open source implementations of OpenGL and more
        libglib2.0-0 \
        libegl1 \
        # Install libegl1 libgl1 libgomp1 for open3d
        libgl1 \
        libgomp1 \
        # install graphical libraries used by qt, vispy and napari  \
        # according to napari dockerfile https://github.com/napari/
        libfontconfig1 \
        libxrender1 \
        libdbus-1-3 \
        libxkbcommon-x11-0 \
        libxi6 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xinput0 \
        libxcb-xfixes0 \
        libxcb-shape0 \
        # install git for setuptools_scm to deal with locan source distribution versioning
        git && \
    # and remove package manager cache
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
    conda install \
    --yes \
    --name base \
    python=$PYTHON_VERSION \
    locan \
    #napari-locan \
    jupyterlab && \
    conda env update -n base --file environment.yml && \
    source activate base && \
# to install from source code use:
#    pip install . && \
    conda clean -afy


FROM base

LABEL maintainer="LocanDevelopers" \
      project="locan" \
      python_version=${PYTHON_VERSION}

COPY --from=build --chown=$USER:$USER /opt/conda /opt/conda

USER $USER

# Volume for data
VOLUME ["/shared"]

WORKDIR "/shared"

RUN jupyter server --generate-config && \
    jupyter lab clean && \
    locan show_versions -e -v

ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

CMD jupyter lab --no-browser --ip 0.0.0.0

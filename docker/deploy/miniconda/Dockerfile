# Dockerfile for deploying surepy, napari and others in a conda environment as non-root user.
#
# This dockerfile is in parts adapted from dockerfiles by the Jupyter Development Team
# https://hub.docker.com/r/jupyter/base-notebook/dockerfile
# and from Continuum Analytics, Inc.
# https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile
#
FROM debian:buster-slim

ARG PYTHON_VERSION=3.8
ARG MINICONDA_VERSION=latest
ARG USER=asterix
ARG UID=1000
ARG GID=100

ENV PYTHON_VERSION=$PYTHON_VERSION \
    ENVIRONMENT_NAME=surepy \
    MINICONDA_VERSION=$MINICONDA_VERSION \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    USER=$USER \
    UID=$UID \
    GID=$GID \
    HOME=/home/$USER \
    CONDA_DIR=/home/$USER/miniconda \
    # path to new conda environment
    PATH=$CONDA_DIR/bin:$PATH \
    PATH=$CONDA_DIR/envs/$ENVIRONMENT_NAME/bin:$PATH \
    # Variables required to install libglib2.0-0 non-interactively (for napari)
    TZ=Europe/Amsterdam \
    DEBIAN_FRONTEND=noninteractive

LABEL maintainer="SurepyDevelopers" \
      project="surepy" \
      python_version=${PYTHON_VERSION}

RUN apt-get update --fix-missing && \
    # set time zone to local time
    ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime && \
    # install libraries for miniconda installer
    apt-get install -qqy wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 && \
    # install GL library (including GLX and DRI)
    apt-get install -qqy libgl1-mesa-dev && \
    # install libraries for napari
    # && apt-get install -qqy libxi6 libglib2.0-0 fontconfig libfontconfig1 libxrender1 libdbus-1-3 \
    apt-get install -qqy libxi6 fontconfig libfontconfig1 libdbus-1-3 && \
    # remove package manager cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Create a non-root user
    adduser --disabled-password \
         --gecos "Non-root user" \
         --uid $UID \
         --gid $GID \
         --home $HOME \
         $USER

#Activate bash shell for correct use of conda commands
SHELL [ "/bin/bash", "--login", "-c" ]

USER $USER

# install miniconda3 according to https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    # make conda activate command available from /bin/bash shells
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    # make conda activate command available from /bin/bash --login shells
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile && \
    # update
    $CONDA_DIR/bin/conda update -n base -c defaults conda && \
    $CONDA_DIR/bin/conda install -y python=$PYTHON_VERSION && \
    $CONDA_DIR/bin/conda update -n base --all && \
    # clean up
    find $CONDA_DIR -follow -type f -name '*.a' -delete && \
    find $CONDA_DIR -follow -type f -name '*.js.map' -delete && \
    $CONDA_DIR/bin/conda clean -afy && \
    rm -rf $HOME/.cache/yarn

# Copy the current directory contents into the container
COPY --chown=$UID:$GID . $HOME/surepy

WORKDIR $HOME/surepy

# create new environment
RUN conda init bash && \
    conda env create -n $ENVIRONMENT_NAME python=$PYTHON_VERSION && \
    conda env update -n $ENVIRONMENT_NAME --file environment.yml && \
    conda activate $ENVIRONMENT_NAME && \
    # install surepy
    python setup.py install && \
    # currently napari scripts require an older version
    conda install -c conda-forge napari=0.2.12 --freeze-installed && \
    # clean up
	echo "conda activate $ENVIRONMENT_NAME" >> ~/.bashrc && \
	rm -rf $HOME/surepy/* && \
	conda clean -afy && \
    rm -rf $HOME/.cache/yarn

# Volume for data
VOLUME ["$HOME/shared"]

WORKDIR $HOME/shared

# Provide a test and environment file for reference:
RUN conda activate $ENVIRONMENT_NAME && \
    mkdir $HOME/surepy/reports && \
    today=$(date +"%Y-%m-%d") && \
    base=${ENVIRONMENT_NAME}_$today && \
    conda env export > "$HOME/surepy/reports/environment_$base.yml" && \
    date > "$HOME/surepy/reports/test_results_$base.txt" && \
    surepy test >> "$HOME/surepy/reports/test_results_$base.txt"

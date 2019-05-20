# Use a Jupyter parent image:
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
FROM jupyter/minimal-notebook

# Set up the current project
LABEL maintainer="SurepyDevelopers" \
      project="surepy"

USER root

# Copy the current directory contents into the container
COPY . /surepy

WORKDIR /surepy

# Install packages specified in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install packages specified in conda environment file
# RUN conda env update --file environment.yml

# install myProject
RUN ["python", "setup.py", "install"]

RUN fix-permissions $CONDA_DIR
RUN fix-permissions /home/$NB_USER

# start in home directory.
WORKDIR /home/$NB_USER/work

USER $NB_UID

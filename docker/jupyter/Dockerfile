# Use an official image for Jupyter that is based on miniconda and ubuntu
FROM jupyter/minimal-notebook

# Set up the current project
LABEL maintainer="SurepyDevelopers" \
      project="surepy"

USER root

# Copy the current directory contents into the container
COPY . /surepy

WORKDIR /surepy

# Update requirements from environment.yml
RUN conda env update -n base -f environment.yml

# Install the project
RUN ["python", "setup.py", "install"]

RUN fix-permissions $CONDA_DIR
RUN fix-permissions /home/$NB_USER

# Start in home directory.
WORKDIR /home/$NB_USER/work

USER $NB_UID

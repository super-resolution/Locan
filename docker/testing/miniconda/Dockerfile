# Use an official image for mambaforge that is based on debian
FROM condaforge/mambaforge:latest

ARG PYTHON_VERSION=3.9
ARG ENVIRONMENT_NAME="locan"

ENV PYTHON_VERSION=${PYTHON_VERSION} \
    ENVIRONMENT_NAME=${ENVIRONMENT_NAME}
    # ENVIRONMENT_NAME is used for naming output files. The conda environment is "locan".

LABEL maintainer="LocanDevelopers" \
      project="locan" \
      python_version=${PYTHON_VERSION} \
      environment_name=${ENVIRONMENT_NAME}
	  
# set time zone to local time
RUN ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# Install GL library and remove package manager cache
RUN apt-get update && \
    apt-get install libgl1-mesa-dev -y && \
    # install git for setuptools_scm to deal with locan source distribution versioning
    apt-get install -qqy git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#Activate bash shell for correct use of conda commands
SHELL [ "/bin/bash", "--login", "-c" ]

# Copy the current directory contents into the container
COPY . locan

WORKDIR /locan

# Update conda & Create environment from environment.yml
RUN mamba init bash && \
    conda update --name base conda mamba && \
    mamba create --name "locan" python=$PYTHON_VERSION && \
    mamba env update --name "locan" --file environment.yml

#Activate bash shell for correct use of conda commands in new environment
SHELL ["conda", "run", "-n", "locan", "/bin/bash", "-c"]

# Install the project and clean up
RUN mamba init bash && \
    pip install . && \
    conda clean -afy

# Volume for data
VOLUME ["/shared"]

# Run a command when the container launches
CMD mamba init bash && \
    today=$(date +"%Y-%m-%d") && \
    base=${ENVIRONMENT_NAME}_$today && \
    echo $PATH > "/shared/path_$base.txt" && \
    conda env export > "/shared/environment_$base.yml" && \
    date > "/shared/versions_$base.txt" && \
    locan show_versions -e -v >> "/shared/versions_$base.txt" && \
    date > "/shared/test_results_$base.txt" && \
    locan test >> "/shared/test_results_$base.txt"

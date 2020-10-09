# Use an official image for miniconda3 that is based on debian
FROM continuumio/miniconda3

ARG PYTHON_VERSION=3.8
ARG ENVIRONMENT_NAME="surepy_conda_py38"

ENV PYTHON_VERSION=${PYTHON_VERSION} \
    ENVIRONMENT_NAME=${ENVIRONMENT_NAME}

LABEL maintainer="SurepyDevelopers" \
      project="surepy" \
      python_version=${PYTHON_VERSION} \
      environment_name=${ENVIRONMENT_NAME}
	  
# set time zone to local time
RUN ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# Install GL library and remove package manager cache
RUN apt-get update \
 && apt-get install libgl1-mesa-dev -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

#Activate bash shell for correct use of conda commands
SHELL [ "/bin/bash", "--login", "-c" ]

# Copy the current directory contents into the container
COPY . surepy

WORKDIR /surepy

# Update conda & Create environment from environment.yml & Install the project
RUN conda init bash \
 && conda update -n base -c defaults conda \
 && conda env create -n $ENVIRONMENT_NAME -f environment.yml python=$PYTHON_VERSION \
 && conda activate $ENVIRONMENT_NAME \
 && python setup.py install \
 && conda clean -afy

# Volume for data
VOLUME ["/shared"]

# Run a command when the container launches
CMD conda activate $ENVIRONMENT_NAME \
 && today=$(date +"%Y-%m-%d") \
 && base=${ENVIRONMENT_NAME}_$today \
 && echo $PATH > "/shared/path_$base.txt" \
 && conda env export > "/shared/environment_$base.yml" \
 && date > "/shared/test_results_$base.txt" \
 && pytest >> "/shared/test_results_$base.txt"
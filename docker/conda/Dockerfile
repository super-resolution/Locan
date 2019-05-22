# Use an official image for miniconda3 that is based on debian
FROM continuumio/miniconda3

LABEL maintainer="SurepyDevelopers" \
      project="surepy"

# Copy the current directory contents into the container
COPY . /surepy

WORKDIR /surepy

# Update requirements from environment.yml
RUN conda env update -n base -f environment.yml

# Install the project
RUN ["python", "setup.py", "install"]

# Run a command when the container launches
CMD pytest

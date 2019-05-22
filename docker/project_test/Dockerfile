# Use an official Linux Python runtime as a parent image
FROM python:3

LABEL maintainer="SurepyDevelopers" \
      project="surepy"

# Copy the current directory contents into the container
COPY . /myProject

WORKDIR /myProject

# Install needed packages
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install the project
RUN ["python", "setup.py", "install"]

# Run a command when the container launches
CMD pytest

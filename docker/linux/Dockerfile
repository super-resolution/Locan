# Use an official image for a Python runtime that is based on ubuntu
FROM python:3

LABEL maintainer="SurepyDevelopers" \
      project="surepy"

# Copy the current directory contents into the container
COPY . /surepy

WORKDIR /surepy

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install the project
RUN ["python", "setup.py", "install"]

# Run a command when the container launches
CMD pytest

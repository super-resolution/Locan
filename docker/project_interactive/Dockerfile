# Use an official Python runtime as a parent image
FROM python:3

LABEL maintainer="SurepyDevelopers" \
      project="surepy"

# Copy the current directory contents into the container
COPY . /myProject

WORKDIR /myProject

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# install myProject
RUN ["python", "setup.py", "install"]

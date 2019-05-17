# Use an official Python runtime as a parent image
FROM python:3

LABEL maintainer="Soeren" \
      project="Surepy"

# Set the working directory to /code
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY . /code

# install code
RUN bash -c python setup.py install;

# Install any needed packages specified in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run app.py when the container launches
CMD ["python setup.py test"]

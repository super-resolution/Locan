# Docker

You can use the provided Dockerfiles to set up a docker image and use this project within a container.

## Prepare for using Docker

1) Install docker
2) Switch to Linux containers
3) Make shared drives on your host system accessible to mount local directories:

    Docker --> Settings --> Shared Drives

## Dockerfile

We provide a Dockerfile for the following tasks:

1) Run the project in a Python 3 environment on Linux.
2) Run the project in a Conda environment on Linux.
3) Provide a Conda environment with Jupyter lab for interactive work.

## Build a docker image

Download the source code in a project directory.
Make sure the .dockerignore file is present.

Enter the project directory and run the following command to build the docker image from one of the Dockerfiles:

	docker build -t <ImageName> -f <docker/choose directory/Dockerfile> .
		
## Start a container from the image:

### Running project tests or starting an interactive environment

Run a container to just run the project tests and close afterwards:

	docker run <ImageName>
	
Open a shell (e.g. bash) for interactive work within a container:

	docker run -it <ImageName> bash
	
Open the shell with a host directory mounted as volume:

	docker run -it -v <host directory>:<container directory> <ImageName> bash
	
### Jupyter notebooks

Run the image to start a container with an interactive shell: 

	docker run -it jupyter_surepy bash
	
Or run the image to start a container with jupyter lab
	
	docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes <ImageName>

Start jupyter lab with a host directory mounted as volume:

	docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v <host/directory>:/home/jovyan/work <ImageName>
	
For the host directory to be your home directory put in "~" .

After closing a jupyter lab you might have to delete the container before running the image again 
(e.g. if the option --rm was omitted):

    docker rm -f <container>

or close all containers:

    docker rm -f $(docker ps -q)
    
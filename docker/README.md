# Docker

You can use the provided Dockerfiles to set up a docker image and run the project within a container.


## Prepare for using Docker

1) Install docker (Docker Desktop)
2) Switch to Linux containers
3) Make shared drives on your host system accessible to mount local directories:

    Docker --> Settings --> Shared Drives


## Dockerfiles

We provide dockerfiles for testing and deployment.

1) Run tests in a Python 3 environment based on miniconda or a virtual environment (venv) on Debian Linux.
2) Run the project in a Python 3 miniconda environment with Jupyter lab for interactive work (Debian Linux).


## Build a docker image

Download the source code in a project directory.
Make sure the .dockerignore file is present.

Enter the project directory and run the following command to build the docker image from one of the Dockerfiles:

```
docker build -t <ImageName> -f <docker/choose directory/Dockerfile> .
```


## Start a container from the image

### Run project tests:

Run a container to just run the project tests and close afterwards:

```
docker run --rm <ImageName>
```

### Run project in an interactive environment:
	
Open a bash shell for interactive work within a container:

```
docker run -it <ImageName> bash
```

Open the shell with a host directory mounted as volume:

```
docker run -it -v <host directory>:/home/asterix/shared <ImageName> bash
```

To make sure a gui output (e.g. from napari) is directed to an X server (that must be installed on your host) 
add the DISPLAY environment variable:

```
docker run -it -v <host directory>:/home/asterix/shared -e DISPLAY=<IP-address>:0.0 <ImageName> bash
```

### Use Jupyter notebooks:
	
Start a container providing browser access to jupyter lab with a host directory mounted as volume:

```	
docker run -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v <host directory>:/home/asterix/shared <ImageName>
```

For gui (e.g. napari) interaction add the DISPLAY environment variable.


## Clean up

After closing a jupyter lab you might have to delete the container before running the image again 
(e.g. if the option --rm was omitted):

```
docker rm -f <container>
```

or close all containers:

```
docker rm -f $(docker ps -q)
```

To clean up your system:

```
docker system prune
```

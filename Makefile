# Makefile for building and running a Docker environment

# Variables
IMAGE_NAME := st_app
CONTAINER_NAME := st_app

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) . --network=host

# Run the Docker container
# docker run -it --name $(CONTAINER_NAME) $(IMAGE_NAME)
run:
	docker run -it -p 8501:8501 --name $(CONTAINER_NAME) $(IMAGE_NAME)

bash:
	docker run --rm -it $(CONTAINER_NAME) bash
# Stop and remove the Docker container
stop:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

# Remove the Docker image
clean:
	docker rmi $(IMAGE_NAME)

# Phony targets
.PHONY: build run bash stop clean

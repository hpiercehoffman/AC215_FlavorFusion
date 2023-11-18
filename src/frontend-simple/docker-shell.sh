#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
# Automatic export to the environment of subsequently executed commands
# source: the command 'help export' run in Terminal
export IMAGE_NAME="test-frontend"
export BASE_DIR=$(pwd)

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .
echo "Changed the port"

# Run the container
# --v: Attach a filesystem volume to the container
# -p: Publish a container's port(s) to the host (host_port: container_port) 
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-p 9000:9000 $IMAGE_NAME
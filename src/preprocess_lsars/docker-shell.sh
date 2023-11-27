#!/bin/bash

set -e

# Create the network if we don't have it yet
docker network inspect preprocess-lsars-network >/dev/null 2>&1 || docker network create preprocess-lsars-network

# Build the image based on the Dockerfile
docker build -t preprocess-lsars --platform=linux/arm64/v8 -f Dockerfile .

# Run All Containers
docker-compose run --rm --service-ports preprocess-lsars

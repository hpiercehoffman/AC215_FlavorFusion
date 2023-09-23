#!/bin/bash

set -e

# Create the network if we don't have it yet
docker network inspect preprocess-google-network >/dev/null 2>&1 || docker network create preprocess-google-network

# Build the image based on the Dockerfile
docker build -t preprocess-google --platform=linux/arm64/v8 -f Dockerfile .

# Run All Containers
docker-compose run --rm --service-ports preprocess-google

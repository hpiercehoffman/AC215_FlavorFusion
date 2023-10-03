#!/bin/bash

set -e

# Create the network if we don't have it yet
docker network inspect train-primera-network >/dev/null 2>&1 || docker network create train-primera-network

# Build the image based on the Dockerfile
docker build -t train-primera --platform=linux/amd64 -f Dockerfile .

# Run All Containers
docker compose run --rm --service-ports train-primera

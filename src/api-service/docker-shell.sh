#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="test-api"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export PERSISTENT_DIR=$(pwd)
export GCS_BUCKET_NAME="reviews-data"

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$PERSISTENT_DIR":/persistent \
-p 9000:9000 \
-e DEV=1 \
-e WANDB_KEY=$(cat /secrets/wandb_key.txt) \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/google_secrets.json \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
--privileged \
$IMAGE_NAME


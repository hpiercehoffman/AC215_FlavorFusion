# We will use the official Debian-hosted Python image.
FROM python:3.9-slim-buster

# Tell pipenv where the shell is. 
# This allows us to use "pipenv shell" as a container entry point.
ENV PYENV_SHELL=/bin/bash

# Authenticate via the flavor-fusion service account. If using 
# gcloud CLI, these credentials are not needed.
ENV GOOGLE_APPLICATION_CREDENTIALS=secrets/flavor-fusion.json

# Ensure we have an up to date baseline and install dependencies. 
RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential git ffmpeg && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    mkdir -p /app

WORKDIR /app

# Add Pipfile, Pipfile.lock
COPY Pipfile Pipfile.lock requirements.txt /app/

RUN pip install -r requirements.txt

# Source code
ADD . /app

# Entry point
ENTRYPOINT ["/bin/bash"]

# Get into the pipenv shell
CMD ["-c", "pipenv shell"]

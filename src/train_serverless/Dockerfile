# We will use the official Debian-hosted Python image.
FROM python:3.9-slim-buster

ARG DEBIAN_PACKAGES="build-essential git curl"
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Tell pipenv where the shell is. 
# This allows us to use "pipenv shell" as a container entry point.
ENV PYENV_SHELL=/bin/bash

# Authenticate via the flavor-fusion service account. If using 
# gcloud CLI, these credentials are not needed.
#ENV GOOGLE_APPLICATION_CREDENTIALS=secrets/flavor-fusion-service.json

# Ensure we have an up to date baseline and install dependencies. 
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 Passw0rd)" && \
    mkdir -p /app && \
    chown app:app /app

USER app
WORKDIR /app

# Add Pipfile, Pipfile.lock
ADD --chown=app:app Pipfile Pipfile.lock /app/
RUN pipenv install
RUN pipenv sync

#RUN pip install -r requirements.txt

# Source code
ADD --chown=app:app . /app
ENV PATH="${PATH}:/home/app/.local/bin"
ENV Path="${PATH}:/home/app/.local/lib/python3.9/site-packages"

# Entry point
ENTRYPOINT ["python","train_serverless.py"]


FROM anibali/pytorch:2.0.1-cuda11.8

USER root
RUN apt-get -y update
RUN apt-get -y install python-pip

# This image already has /app directory created, so we don't need to create it
WORKDIR /app
ADD . /app

RUN cd /app
RUN pip install -r requirements.txt

# Entry point
ENTRYPOINT ["python","finetune_serverless.py"]


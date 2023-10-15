FROM anibali/pytorch:2.0.1-cuda11.8

USER root
RUN apt-get -y update
RUN apt-get -y install python-pip
#RUN apt-get install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget

#RUN mkdir /app
WORKDIR /app
ADD . /app

RUN cd /app
RUN pip install -r requirements.txt

# Entry point
ENTRYPOINT ["python","train_serverless.py"]
#ENTRYPOINT ["/bin/bash"]


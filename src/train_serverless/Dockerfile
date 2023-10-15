FROM nvidia/cuda:11.0.3-devel-centos7

RUN yum -y update
RUN yum -y install epel-release
RUN yum -y install python-pip
RUN yum install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget
RUN wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz

RUN mkdir /app
WORKDIR /app
ADD . /app

RUN tar -xvf Python-3.9.6.tgz
RUN cd Python-3.9.6 && ./configure --enable-optimizations
RUN sudo make && sudo make altinstall

RUN cd /app
RUN pip install -r requirements.txt

# Entry point
ENTRYPOINT ["python","train_serverless.py"]


FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

WORKDIR /training

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN rm /etc/apt/sources.list.d/cuda.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    	python3 \
    	python-is-python3 \
    	python3-setuptools \
    	python3-wheel \
    	python3-pip \
    	curl

# Install GCloud CLI
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg  \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - \
    && apt-get update -y \
    && apt-get install google-cloud-cli -y


ADD . .


RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

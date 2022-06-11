FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

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
    	python3-pip

ADD . .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

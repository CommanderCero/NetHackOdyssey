FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04 AS base

WORKDIR /workspace

RUN apt update && apt-get -y install \
    clang \
    curl \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python-is-python3 \
    sudo

# NLE Dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install \
    build-essential \
    autoconf \
    libtool \
    pkg-config \
    flex \
    bison \
    libbz2-dev \
    wget \
    software-properties-common \
    cmake \
    freeglut3-dev

# OpenCV Dependencies
#RUN apt-get -y install \
#    ffmpeg \
#    libsm6 \
#    libxext6

# Install requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

# Install our project in developer mode
COPY setup.py .
COPY odyssey .
RUN pip install -e .
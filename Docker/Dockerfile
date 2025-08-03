FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

LABEL version="1.0" maintainer="Unknownue <unknownue@outlook.com>"

ARG DEBIAN_FRONTEND=noninteractive

# ADD mirror-ubuntu1604.txt /etc/apt/sources.list

RUN apt update && \
    apt install -y software-properties-common && \
    apt install -y wget git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    add-apt-repository ppa:ubuntu-toolchain-r/test

# Install python 3.9 and pip (copy get-pip.py from build context)
RUN apt update && \
    apt install -y python3.9 python3.9-tk python3.9-distutils

COPY get-pip.py /tmp/get-pip.py

RUN python3.9 /tmp/get-pip.py && rm /tmp/get-pip.py && \
    ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# RUN pip install pqi && \
#     pqi use aliyun

# Install tensorflow
RUN pip install --no-cache-dir tensorflow-gpu==2.10.0
RUN pip install --no-cache-dir numpy==1.21.6 matplotlib==3.5.3 scikit-learn==1.0.2 && \
    pip install --no-cache-dir plyfile Pillow tqdm

RUN pip install --no-cache-dir open3d==0.14.1 && \
    apt install -y libgl1-mesa-glx python-apt && \
    apt update && \
    apt upgrade -y libstdc++6 && \
    rm /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python

# Thêm dòng này để cài laspy
RUN pip install --no-cache-dir laspy

CMD ["bash"]
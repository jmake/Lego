FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    vim \
    wget \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-distutils

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

#RUN apt-get install -y cmake cmake-curses-gui build-essential 

CMD ["bash"]

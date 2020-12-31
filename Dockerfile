FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Install Caffe 1.0
# RUN conda install -c conda-forge caffe==1.0
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        vim \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.6
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.6.9 \
 && conda clean -ya

ENV CAFFE_ROOT=/home/user
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
ENV CLONE_TAG=1.0

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git && cd caffe && \
    pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && sudo make -j install && cd .. && sudo rm -rf nccl

# Fix the libboost to python 3
RUN cd /usr/lib/x86_64-linux-gnu \
    && sudo unlink libboost_python.so \
    && sudo unlink libboost_python.a \
    && sudo ln -s libboost_python-py35.so libboost_python.so \
    && sudo ln -s libboost_python-py35.a libboost_python.a

# Fix the Makefile for CUDA 9 and install Caffe
RUN cd caffe && mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="50 52 60 61" -D python_version=3 .. && \
    sudo make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/caffe/python
ENV PYTHONPATH $PYCAFFE_ROOT:/home/user/miniconda/bin/python
ENV PATH $CAFFE_ROOT/caffe/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/caffe/build/lib" | sudo tee -a /etc/ld.so.conf.d/caffe.conf && sudo ldconfig

# CUDA 9.0-specific steps
RUN conda install pytorch==1.0.0 torchvision==0.2.1 cuda90 -c pytorch

# Install OpenCV 3.4.1
RUN conda install -c conda-forge opencv==3.4.1

# Install additional Utils
RUN pip install setproctitle imutils opencv-contrib-python && pip install python-dateutil --force-reinstall --upgrade

# fix the libx264 issue
RUN conda install ffmpeg x264=20131218 -c conda-forge && conda update ffmpeg

WORKDIR /app

COPY . .

RUN wget -O goturn/nets/tracker.caffemodel http://cs.stanford.edu/people/davheld/public/GOTURN/weights_init/tracker_init.caffemodel

# Set the default command to python3
CMD ["/bin/bash"]

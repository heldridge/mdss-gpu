FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

ARG NTL_VERSION=11.5.1
ARG NTL_URL=https://libntl.org/ntl-${NTL_VERSION}.tar.gz
ARG NTL_SHA256=210d06c31306cbc6eaf6814453c56c776d9d8e8df36d74eb306f6a523d1c6a8a


ARG GMP_VERSION=6.3.0
ARG GMP_URL=https://gmplib.org/download/gmp/gmp-${GMP_VERSION}.tar.xz
ARG GMP_SHA256=a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898


# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install tools
RUN apt-get update && apt-get install -y \
    cmake\
    build-essential \
    m4 \
    git \
    curl \
    wget \
    ca-certificates\
    xz-utils\
    libglib2.0-dev \
    vim \
    htop \
    nvtop \
    libgmp-dev \
    magic-wormhole \
    time

RUN rm -rf /var/lib/apt/lists/* 

# ============================================================
# 2. Build and install NTL 11.5.1
# ============================================================
WORKDIR /tmp/ntl

RUN wget -O ntl.tar.gz "${NTL_URL}" \
 && echo "${NTL_SHA256}  ntl.tar.gz" | sha256sum -c - \
 && tar -xf ntl.tar.gz \
 && cd ntl-${NTL_VERSION}/src \
 && ./configure NTL_GMP_LIP=on \
 && make -j"$(nproc)" \
 && make install \
 && ldconfig \
 && cd / \
 && rm -rf /tmp/ntl


# ============================================================
# Copy in GPU decoder code
# ============================================================

COPY ./gpu_decoder /workspace/gpu_decoder


# ============================================================
# Compile decoder
# ============================================================

WORKDIR /workspace/gpu_decoder/build/

RUN cmake .. -DBUILD_TESTING=FALSE -DBUILD_ALL_ARCHS=ON \
 && make

# Set default working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]


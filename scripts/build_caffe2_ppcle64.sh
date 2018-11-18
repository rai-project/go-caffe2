#!/bin/sh

FRAMEWORK_VERSION=v1.0rc1
CAFFE2_SRC_DIR=$HOME/code/pytorch
CAFFE2_DIST_DIR=/opt/pytorch/caffe2

if [ ! -d "$CAFFE2_SRC_DIR" ]; then
  git clone --single-branch --depth=1 --recurse-submodules --branch=$FRAMEWORK_VERSION https://github.com/pytorch/pytorch.git $CAFFE2_SRC_DIR
fi

if [ ! -d "$CAFFE2_DIST_DIR" ]; then
mkdir -p $CAFFE2_DIST_DIR
fi

cd $CAFFE2_SRC_DIR && git submodule update --init && mkdir -p build && cd build && \
  cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CAFFE2_DIST_DIR \
    -DUSE_OBSERVERS=ON \
    -DBLAS=OpenBLAS \
    -DBUILD_CUSTOM_PROTOBUF=OFF \
    -DBUILD_PYTHON=ON \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    -DUSE_OPENCV=ON \
    -DUSE_GLOO=OFF \
    -DUSE_NCCL=OFF \
    -DUSE_CUDA=ON \
    -DUSE_CUDNN=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DTORCH_CUDA_ARCH_LIST="3.0 3.5 5.0 5.2 6.0 6.1+PTX 7.0+PTX" \
    -DPYTORCH_CUDA_ARCH_LIST="3.0 3.5 5.0 5.2 6.0 6.1+PTX 7.0+PTX" \

  make -j"$(nproc)" install


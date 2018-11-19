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

cd $CAFFE2_SRC_DIR && git submodule update --init && rm -rf build && mkdir -p build && cd build && \
	cmake .. \
    -DBUILD_PYTHON=ON \
    -DBUILD_BINARY=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILDING_WITH_TORCH_LIBS=OFF \
    -DBUILD_CUSTOM_PROTOBUF=OFF \
	  -DUSE_OBSERVERS=ON \
    -DUSE_NATIVE_ARCH=ON \
    -DBLAS=OpenBLAS \
    -DUSE_NNPACK=ON \
    -DUSE_ROCKSDB=OFF \
    -DUSE_OPENCV=ON \
    -DUSE_DISTRIBUTED=OFF \
    -DUSE_NCCL=OFF \
    -DUSE_PROF=ON \
    -DUSE_MKL=OFF \
    -DCMAKE_INSTALL_PREFIX=$CAFFE2_DIST_DIR \

make -j4 install


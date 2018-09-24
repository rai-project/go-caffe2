FRAMEWORK_VERSION=v0.4.1
CAFFE2_SRC_DIR=$HOME/code/pytorch

git clone --single-branch --depth=1 --recurse-submodules --branch=$FRAMEWORK_VERSION https://github.com/pytorch/pytorch.git $CAFFE2_SRC_DIR

CAFFE2_DIST_DIR=/opt/pytorch/caffe2
mkdir -p $CAFFE2_DIST_DIR

cd $CAFFE2_SRC_DIR && git submodule update --init && mkdir -p build && cd build && \
  cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CAFFE2_DIST_DIR \
    -DUSE_OBSERVERS=ON \
    -DBLAS=OpenBLAS \
    -DBUILD_CUSTOM_PROTOBUF=OFF \
    -DBUILD_PYTHON=OFF \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    -DUSE_OPENCV=OFF \
    -DUSE_GLOO=OFF \
    -DUSE_NCCL=OFF \
    -DUSE_CUDA=ON \
    -DUSE_CUDNN=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="35 52 60 61 70" \
    -DCUDA_ARCH_PTX="61 70" && \
	make -j"$(nproc)" install


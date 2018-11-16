FRAMEWORK_VERSION=master
CAFFE2_SRC_DIR=$HOME/code/pytorch

git clone --single-branch --depth=1 --recurse-submodules --branch=$FRAMEWORK_VERSION https://github.com/pytorch/pytorch.git $CAFFE2_SRC_DIR

CAFFE2_DIST_DIR=/opt/pytorch/caffe2
mkdir -p $CAFFE2_DIST_DIR

cd $CAFFE2_SRC_DIR && git submodule update --init && mkdir -p build && cd build && \
	cmake .. \
	  -DUSE_OBSERVERS=ON \
      -DUSE_NATIVE_ARCH=ON \
      -DBLAS=OpenBLAS \
      -DUSE_CUDA=ON \
      -DUSE_CUDNN=ON \
      -DUSE_NNPACK=OFF \
      -DUSE_ROCKSDB=OFF \
      -DBUILD_PYTHON=OFF \
      -DBUILD_BINARY=OFF \
      -DUSE_OPENCV=OFF \
      -DUSE_NNPACK=OFF \
      -DUSE_GLOO=OFF \
      -DUSE_NCCL=OFF \
      -DUSE_PROF=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILDING_WITH_TORCH_LIBS=OFF \
      -DBUILD_CUSTOM_PROTOBUF=OFF \
      -DTORCH_CUDA_ARCH_LIST="3.0 3.5 5.0 5.2 6.0 6.1+PTX 7.0+PTX" \
      -DPYTORCH_CUDA_ARCH_LIST="3.0 3.5 5.0 5.2 6.0 6.1+PTX 7.0+PTX" \
      -DCMAKE_INSTALL_PREFIX=$CAFFE2_DIST_DIR \
	&& make -j"$(nproc)" install


FRAMEWORK_VERSION=v0.4.1
CAFFE2_SRC_DIR=$HOME/code/pytorch

git clone --single-branch --depth=1 --recurse-submodules -j8 --branch=$FRAMEWORK_VERSION https://github.com/pytorch/pytorch.git $CAFFE2_SRC_DIR

DIST_DIR=/opt/pytorch/caffe2
mkdir -p $DIST_DIR

cd $CAFFE2_SRC_DIR && git submodule update --init && mkdir -p build && cd build && \
	cmake .. \
		 -DUSE_OBSERVERS=ON \
      -DBLAS=OpenBLAS \
      -DUSE_CUDA=OFF \
      -DUSE_NNPACK=OFF \
      -DUSE_ROCKSDB=OFF \
      -DBUILD_PYTHON=OFF \
      -DUSE_OPENCV=OFF \
      -DUSE_NNPACK=OFF \
      -DUSE_GLOO=OFF \
      -DUSE_NCCL=OFF \
      -DUSE_PROF=ON \
      -DBUILD_CUSTOM_PROTOBUF=OFF \
      -DCMAKE_INSTALL_PREFIX=$DIST_DIR \
	&& make -j4 install


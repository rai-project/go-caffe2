FRAMEWORK_VERSION=v0.8.1
CUB_VERSION=1.7.3
CAFFE2_SRC_DIR=$HOME/code/caffe2

git clone --single-branch --branch $FRAMEWORK_VERSION https://github.com/caffe2/caffe2.git $CAFFE2_SRC_DIR

cd $CAFFE2_SRC_DIR && \
	git submodule init && \
    	git rm third_party/cub && \
    	cd third_party && \
    	git clone --branch $CUB_VERSION https://github.com/NVlabs/cub

cd $CAFFE2_SRC_DIR && \
	git submodule update --init --recursive && \
    	cd third_party/benchmark && \
    	git checkout master && \
    	git pull

DIST_DIR=$HOME/frameworks/caffe2
mkdir -p $DIST_DIR

cd $CAFFE2_SRC_DIR && mkdir build && cd build && \
	cmake .. \
		-DCMAKE_INSTALL_PREFIX=$DIST_DIR \
		-DUSE_CUDA=1 \
		-DUSE_NCCL=1 \
		-DBUILD_SHARED_LIBS=1 \
		-DNCCL_INCLUDE_DIR=/opt/DL/nccl/include \
		-DNCCL_LIBRARY=/opt/DL/nccl/lib \
		-DCUDA_ARCH_NAME=Manual \
		-DCUDA_ARCH_BIN="35 52 60 61" \
		-DCUDA_ARCH_PTX="61" \
		-DUSE_NNPACK=OFF \
		-DUSE_ROCKSDB=OFF \
		-DUSE_OPENCV=OFF \
		-DBUILD_PYTHON=OFF \
	&& make -j"$(nproc)" install


package caffe2

// #cgo LDFLAGS: -lstdc++ -lglog -lboost_system -lcaffe2
// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function
// #cgo CXXFLAGS: -I/opt/pytorch/caffe2/include -I/usr/include/eigen3/ -I/usr/local/include/eigen3/
// #cgo LDFLAGS: -L/opt/pytorch/caffe2/lib
// #cgo linux CXXFLAGS: -DWITH_GPU=1 -DWITH_CUDA=1 -I/usr/local/cuda/include
// #cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -lcaffe2_gpu -lcurand -lcudart
// #cgo linux CPPFLAGS: -DWITH_GPU=1 -DWITH_CUDA=1
// #cgo pkg-config: protobuf
import "C"

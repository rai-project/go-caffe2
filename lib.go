package caffe2

// #cgo LDFLAGS: -lstdc++ -lglog -lboost_system -lCaffe2_CPU
// #cgo CXXFLAGS:  -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -I/usr/include/eigen3/ -I/usr/local/include/eigen3/
// #cgo darwin CXXFLAGS: -I/opt/caffe2/include -I/opt/pytorch/caffe2/include
// #cgo darwin LDFLAGS: -L/opt/caffe2/lib -L/opt/pytorch/caffe2/lib
// #cgo linux,ppc64le CXXFLAGS: -I/opt/caffe2/include -I/opt/pytorch/caffe2/include -I/home/carml/frameworks/pytorch/caffe2/include -DWITH_GPU=1 -DWITH_CUDA=1 -I/usr/local/cuda/include
// #cgo linux,ppc64le LDFLAGS: -L/opt/caffe2/lib -L/opt/pytorch/caffe2/lib -L/home/carml/frameworks/pytorch/caffe2/lib -L/usr/local/cuda/lib64 -lCaffe2_GPU -lcurand -lcudart
// #cgo linux,amd64 linux,arm64 CXXFLAGS: -I/opt/caffe2/include -I/opt/pytorch/caffe2/include  -I/opt/frameworks/pytorch/caffe2/include -DWITH_GPU=1 -DWITH_CUDA=1 -I/usr/local/cuda/include
// #cgo linux,amd64 linux,arm64 LDFLAGS: -L/opt/caffe2/lib -L/opt/frameworks/pytorch/caffe2/lib -L/usr/local/cuda/lib64 -lCaffe2_GPU -lcurand -lcudart
// #cgo pkg-config: --cflags --libs protobuf
import "C"

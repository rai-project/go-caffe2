package caffe2

// #cgo LDFLAGS: -lstdc++ -lglog -lboost_system -lCaffe2_CPU
// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -I/usr/include/eigen3/ -I/usr/local/include/eigen3/
// #cgo darwin CXXFLAGS: -I/opt/caffe2/include  -I/usr/local/include/eigen3/
// #cgo darwin LDFLAGS: -L/opt/caffe2/lib
// #cgo ppc64le CXXFLAGS: -I/home/carml/frameworks/caffe2/include -DWITH_CUDA=1 -I /usr/local/cuda/include
// #cgo ppc64le LDFLAGS: -L/home/carml/frameworks/caffe2/lib -lCaffe2_GPU
// #cgo amd64 CXXFLAGS: -I/home/abduld/frameworks/caffe2/include -DWITH_CUDA=1 -I /usr/local/cuda/include
// #cgo amd64 LDFLAGS: -L/home/abduld/frameworks/caffe2/lib -lCaffe2_GPU
// #cgo pkg-config: protobuf
import "C"

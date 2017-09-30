package caffe2

// #cgo LDFLAGS: -lstdc++ -lglog -lboost_system
// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -I/usr/include/eigen3/ -I/usr/local/include/eigen3/
// #cgo darwin CXXFLAGS: -I/opt/caffe2/include  -I/usr/local/include/eigen3/
// #cgo darwin LDFLAGS: -L/opt/caffe2/lib -lCaffe2_CPU
// #cgo ppc64le CXXFLAGS: -I/home/carml/frameworks/caffe2/include
// #cgo ppc64le LDFLAGS: -L/home/carml/frameworks/caffe2/lib -lCaffe2_GPU
// #cgo pkg-config: protobuf
import "C"

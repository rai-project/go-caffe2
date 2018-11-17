package caffe2

// #cgo LDFLAGS: -lstdc++ -lglog -lboost_system -lc10 -lsleef -lonnx -lcaffe2 -lcaffe2_observers -L/opt/pytorch/caffe2/lib
// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -Wall -march=native -O3 -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function
// #cgo CXXFLAGS: -I/opt/pytorch/caffe2/include -I/usr/include/eigen3/ -I/usr/local/include/eigen3/ -DBLAS=open -DUSE_OBSERVERS=1
// #cgo darwin CXXFLAGS: -I/usr/local/opt/openblas/include
// #cgo !darwin,!nogpu CXXFLAGS: -DWITH_CUDA=1 -I/usr/local/cuda/include
// #cgo !darwin,!nogpu LDFLAGS: -L/usr/local/cuda/lib64 -lcaffe2_gpu -lcurand -lcudart
// #cgo pkg-config: protobuf
import "C"

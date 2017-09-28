package caffe2

// #cgo LDFLAGS: -lstdc++ -lglog -lboost_system  
// #cgo CXXFLAGS: -DBLAS=open -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -DCPU_ONLY=1 
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -I/usr/include/eigen3/
// #cgo darwin CXXFLAGS: -I/opt/caffe2/include
// #cgo darwin LDFLAGS: -L/opt/caffe2/lib -lcaffe2 
// #cgo ppc64le CXXFLAGS: -I/home/carml/frameworks/caffe2/include
// #cgo ppc64le LDFLAGS: -L/home/carml/frameworks/caffe2/lib -lCaffe2_GPU
import "C"


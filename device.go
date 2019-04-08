package caffe2

// #include "cbits/predictor.hpp"
import "C"

type DeviceKind C.Caffe2_DeviceKind

const (
	UnknownDeviceKind DeviceKind = C.UNKNOWN_DEVICE_KIND
	CPUDeviceKind     DeviceKind = C.CPU_DEVICE_KIND
	CUDADeviceKind    DeviceKind = C.CUDA_DEVICE_KIND
)

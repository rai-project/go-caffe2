package caffe2

// #include "cbits/predictor.hpp"
import "C"
import "reflect"

// DType tensor scalar data type
type DType C.Caffe2_DataType

const (
	UnknownType DType = C.Caffe2_Unknown
	// Byte byte tensors (go type uint8)
	Byte DType = C.Caffe2_Byte
	// Char char tensor (go type int8)
	Char DType = C.Caffe2_Char
	// Int int tensor (go type int32)
	Int DType = C.Caffe2_Int
	// Long long tensor (go type int64)
	Long DType = C.Caffe2_Long
	// Float tensor (go type float32)
	Float DType = C.Caffe2_Float
	// Double tensor  (go type float64)
	Double DType = C.Caffe2_Double
)

var types = []struct {
	typ      reflect.Type
	dataType C.Caffe2_DataType
}{
	{reflect.TypeOf(uint8(0)), C.Caffe2_Byte},
	{reflect.TypeOf(int8(0)), C.Caffe2_Char},
	// {reflect.TypeOf(int16(0)), C.Caffe2_Short},
	{reflect.TypeOf(int32(0)), C.Caffe2_Int},
	{reflect.TypeOf(int64(0)), C.Caffe2_Long},
	// {reflect.TypeOf(float16(0)), C.Caffe2_Half},
	{reflect.TypeOf(float32(0)), C.Caffe2_Float},
	{reflect.TypeOf(float64(0)), C.Caffe2_Double},
}

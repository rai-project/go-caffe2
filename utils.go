package caffe2

import (
	"unsafe"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

func prod(arry []int) int {
	accum := int(1)
	for _, e := range arry {
		accum *= int(e)
	}
	return accum
}

func toTensor(ptr unsafe.Ptr, shape []int, ty tensor.Dtype) (tensor.Tensor, error) {
	flattenedLength := prod(shape)
	switch ty {
	case tensor.Byte, tensor.Uint8:
		{
			cData := (*[1 << 30]uint8)(ptr)[:flattenedLength:flattenedLength]
			data := make([]uint8, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Byte,
				shape,
				tensor.WithBacking(data),
			), nil
		}
	case tensor.Int:
		{
			cData := (*[1 << 30]int)(ptr)[:flattenedLength:flattenedLength]
			data := make([]int, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Int,
				shape,
				tensor.WithBacking(data),
			), nil
		}
	case tensor.Int32:
		{
			cData := (*[1 << 30]int32)(ptr)[:flattenedLength:flattenedLength]
			data := make([]int32, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Int32,
				shape,
				tensor.WithBacking(data),
			), nil
		}
	case tensor.Int64:
		{
			cData := (*[1 << 30]int64)(ptr)[:flattenedLength:flattenedLength]
			data := make([]int64, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Int64,
				shape,
				tensor.WithBacking(data),
			), nil
		}
	case tensor.Float32:
		{
			cData := (*[1 << 30]float32)(ptr)[:flattenedLength:flattenedLength]
			data := make([]float32, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Float32,
				shape,
				tensor.WithBacking(data),
			), nil
		}
	case tensor.Float64:
		{
			cData := (*[1 << 30]float64)(ptr)[:flattenedLength:flattenedLength]
			data := make([]float64, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Float64,
				shape,
				tensor.WithBacking(data),
			), nil
		}
	}

	panic("invalid data type")
	return nil, errors.New("invalid datatype")

}

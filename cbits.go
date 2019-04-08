package caffe2

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"context"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"runtime"
	"unsafe"

	"github.com/rai-project/dlframework/framework/options"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	"gorgonia.org/tensor"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
)

type Device int

const (
	CPUDevice  Device = Device(C.CPU_DEVICE_KIND)
	CUDADevice        = Device(C.CUDA_DEVICE_KIND)
)

type Predictor struct {
	handle  C.PredictorContext
	options *options.Options
}

func New(ctx context.Context, opts ...options.Option) (*Predictor, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_new")
	defer span.Finish()

	options := options.New(opts...)
	initNetFile := string(options.Weights())
	if !com.IsFile(initNetFile) {
		return nil, errors.Errorf("file %s not found", initNetFile)
	}

	isOnnxFormat := filepath.Ext(initNetFile) == ".onnx"
	var predictNetFile string

	if !isOnnxFormat {
		predictNetFile = string(options.Graph())
		if !com.IsFile(predictNetFile) {
			return nil, errors.Errorf("file %s not found", predictNetFile)
		}
	}

	device := C.Caffe2_DeviceKind(CPUDevice)
	if options.UsesGPU() {
		if !nvidiasmi.HasGPU {
			return nil, errors.New("no GPU device")
		}
		device = C.Caffe2_DeviceKind(CUDADevice)
	}

	C.InitCaffe2(device)

	var pred C.PredictorContext
	if isOnnxFormat {
		bts, err := ioutil.ReadFile(initNetFile)
		if err != nil {
			return nil, errors.Wrapf(err, "cannot read %s", initNetFile)
		}

		cNetData := C.CBytes(bts)
		defer func() {
			C.free(unsafe.Pointer(cNetData))
		}()
		pred = C.NewCaffe2FromOnnx(
			(*C.char)(cNetData),
			C.int64_t(len(bts)),
			device,
		)
	} else {
		cInitNetFile := C.CString(initNetFile)
		cPredictNetFile := C.CString(predictNetFile)
		defer func() {
			C.free(unsafe.Pointer(cInitNetFile))
			C.free(unsafe.Pointer(cPredictNetFile))
		}()
		pred = C.NewCaffe2(
			cInitNetFile,
			cPredictNetFile,
			device,
		)
	}

	if pred == nil {
		log.Panicln("unable to create caffe2 predictor")
	}

	p := &Predictor{
		handle:  pred,
		options: options,
	}

	runtime.SetFinalizer(p, (*Predictor).finalize)

	return p, nil
}

func (p *Predictor) AddInput(ctx context.Context, idx int, data tensor.Tensor) error {
}

func (p *Predictor) Predict(ctx context.Context, inputs []tensor.Tensor) error {
	if len(data) == 0 {
		return fmt.Errorf("intput data nil or empty")
	}

	for ii, input := range inputs {
		err := p.AddInput(ctx, ii, input)
		if err != nil {
			return err
		}
	}

	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer span.Finish()

	ok := C.PredictCaffe2(p.handle)
	if ok != 0 {
		return errors.New("unable to perform caffe2 prediction")
	}

	return nil
}

func (p *Predictor) ReadPredictionOutputAtIndex(ctx context.Context, index int) (tensor.Tensor, error) {
	node := p.options.OutputNodes()[index]

	if node.Dtype != tensor.Float32 {
		panic("only supports float32 for now")
	}

	ptr := C.GetPredictionsCaffe2(p.handle, indx)

	shape := node.Shape
	return toTensor(ptr, shape, node.Dtype)
}

func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]tensor.Tensor, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_prediction_output")
	defer span.Finish()

	outputNodes := p.options.OutputNodes()
	res := make([]tensor.Tensor, len(outputNodes))

	for ii := 0; ii < len(outputNodes); ii++ {
		tensor, err := p.ReadPredictionOutputAtIndex(ctx, ii)
		if err != nil {
			return nil, err
		}
		res[ii] = tensor
	}

	return res, nil
}

func (p *Predictor) finalize() {
	if p == nil {
		return
	}
	C.DeleteCaffe2(p.handle)
}

func (p *Predictor) Close() {
	p.finalize()
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.StartProfilingCaffe2(p.handle, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.EndProfilingCaffe2(p.handle)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.DisableProfilingCaffe2(p.handle)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.ReadProfileCaffe2(p.handle)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}

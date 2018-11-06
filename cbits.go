package caffe2

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"context"
	"fmt"
	"unsafe"

	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
)

type Device int

const (
	CPUDevice  Device = Device(C.CPU_DEVICE_KIND)
	CUDADevice        = Device(C.CUDA_DEVICE_KIND)
)

type Predictor struct {
	ctx     C.PredictorContext
	options *options.Options
}

func New(ctx context.Context, opts ...options.Option) (*Predictor, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_new")
	defer span.Finish()

	options := options.New(opts...)
	initNetFile := string(options.Graph())
	if !com.IsFile(initNetFile) {
		return nil, errors.Errorf("file %s not found", initNetFile)
	}
	predictNetFile := string(options.Weights())
	if !com.IsFile(predictNetFile) {
		return nil, errors.Errorf("file %s not found", predictNetFile)
	}

	device := C.DeviceKind(CPUDevice)
	if options.UsesGPU() {
		if !nvidiasmi.HasGPU {
			return nil, errors.New("no GPU device")
		}
		device = C.DeviceKind(CUDADevice)
	}

	C.InitCaffe2(device)

	pred := C.NewCaffe2(C.CString(initNetFile),
		C.CString(predictNetFile),
		device,
	)

	if pred == nil {
		log.Panicln("unable to create caffe2 predictor")
	}

	return &Predictor{
		ctx:     pred,
		options: options,
	}, nil
}

func (p *Predictor) Predict(ctx context.Context, data []float32, channels int,
	width int, height int) error {
	if data == nil || len(data) < 1 {
		return fmt.Errorf("intput data nil or empty")
	}

	batchSize := p.options.BatchSize()
	dataLen := len(data)
	shapeLen := int(width * height * channels)

	inputCount := dataLen / shapeLen
	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))

	predictSpan, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer predictSpan.Finish()

	C.PredictCaffe2(p.ctx, ptr, C.int(batchSize), C.int(channels), C.int(width), C.int(height))

	return nil
}

func (p *Predictor) ReadPredictedFeatures(ctx context.Context) Predictions {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "read_predicted_features")
	defer span.Finish()

	batchSize := p.options.BatchSize()
	predLen := int(C.GetPredLenCaffe2(p.ctx))
	length := batchSize * predLen

	cPredictions := C.GetPredictionsCaffe2(p.ctx)
	defer C.free(unsafe.Pointer(cPredictions))

	slice := (*[1 << 30]C.float)(unsafe.Pointer(cPredictions))[:length:length]

	predictions := make([]Prediction, length)
	for ii := 0; ii < length; ii++ {
		predictions[ii] = Prediction{
			Index:       ii % predLen,
			Probability: float32(slice[ii]),
		}
	}
	return predictions
}

func (p *Predictor) Close() {
	C.DeleteCaffe2(p.ctx)
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.StartProfilingCaffe2(p.ctx, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.EndProfilingCaffe2(p.ctx)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.DisableProfilingCaffe2(p.ctx)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.ReadProfileCaffe2(p.ctx)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}

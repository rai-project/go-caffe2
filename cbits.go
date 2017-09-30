package caffe2

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predict.hpp"
import "C"
import (
	"encoding/json"
	"fmt"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
)

type Predictor struct {
	ctx C.PredictorContext
}

func New(initNetFile, predictNetFile string) (*Predictor, error) {
	if !com.IsFile(initNetFile) {
		return nil, errors.Errorf("file %s not found", initNetFile)
	}
	if !com.IsFile(predictNetFile) {
		return nil, errors.Errorf("file %s not found", predictNetFile)
	}
	ctx := C.New(C.CString(initNetFile), C.CString(predictNetFile))
	if ctx == nil {
		return nil, errors.New("unable to create caffe2 predictor context")
	}
	return &Predictor{
		ctx: ctx,
	}, nil
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.StartProfiling(p.ctx, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.EndProfiling(p.ctx)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.DisableProfiling(p.ctx)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.ReadProfile(p.ctx)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}

func (p *Predictor) Predict(data []float32, batchSize int, channels int,
	width int, height int) (Predictions, error) {
	// check input
	if data == nil || len(data) < 1 {
		return nil, fmt.Errorf("intput data nil or empty")
	}

	if batchSize != 1 {
		dataLen := int64(len(data))
		shapeLen := int64(width * height * channels)
		inputCount := dataLen / shapeLen
		padding := make([]float32, (int64(batchSize)-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))
	r := C.Predict(p.ctx, ptr, C.int(batchSize), C.int(channels), C.int(width), C.int(height))
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	predictions := []Prediction{}
	err := json.Unmarshal([]byte(js), &predictions)
	if err != nil {
		return nil, err
	}
	return predictions, nil
}

func (p *Predictor) Close() {
	C.Delete(p.ctx)
}

func init() {
	C.Init()
}

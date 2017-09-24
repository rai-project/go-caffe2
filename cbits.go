package caffe2

// #cgo LDFLAGS: -lCaffe2_CPU -lstdc++ -lglog -L/usr/local/lib  -L/opt/caffe2/lib
// #cgo CXXFLAGS: -std=c++11 -I/usr/local/include/ -I/usr/include/eigen3/ -I/usr/local/include/eigen3/ -I${SRCDIR}/cbits -O3 -Wall -DCPU_ONLY=1 -I/opt/caffe2/include
// #cgo darwin CXXFLAGS: -DBLAS=open -I/usr/local/opt/openblas/include
// #cgo darwin LDFLAGS: -L/usr/local/opt/openblas/lib
// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predict.hpp"
import "C"
import (
	"encoding/json"
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

func (p *Predictor) Predict(imageData []float32, batch int, channels int,
	width int, height int) (Predictions, error) {
	ptr := (*C.float)(unsafe.Pointer(&imageData[0]))
	r := C.Predict(p.ctx, ptr, C.int(batch), C.int(channels), C.int(width), C.int(height))
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

package main

// #cgo linux CFLAGS: -I/usr/local/cuda/include
// #cgo linux LDFLAGS: -lcuda -lcudart -L/usr/local/cuda/lib64
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_profiler_api.h>
import "C"

import (
	"bufio"
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/downloadmanager"
	caffe2 "github.com/rai-project/go-caffe2"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
)

var (
	batchSize    = 64
	graph_url    = "http://s3.amazonaws.com/store.carml.org/models/caffe2/bvlc_alexnet_1.0/predict_net.pb"
	weights_url  = "http://s3.amazonaws.com/store.carml.org/models/caffe2/bvlc_alexnet_1.0/init_net.pb"
	features_url = "http://data.dmlc.ml/mxnet/models/imagenet/synset.txt"
)

// convert go Image to 1-dim array
func cvtImageTo1DArray(src image.Image, mean []float32) ([]float32, error) {
	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	b := src.Bounds()
	h := b.Max.Y - b.Min.Y // image height
	w := b.Max.X - b.Min.X // image width

	res := make([]float32, 3*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := src.At(x+b.Min.X, y+b.Min.Y).RGBA()
			res[y*w+x] = float32(b>>8) - mean[0]
			res[w*h+y*w+x] = float32(g>>8) - mean[1]
			res[2*w*h+y*w+x] = float32(r>>8) - mean[2]
		}
	}

	return res, nil
}

func main() {
	dir, _ := filepath.Abs("../tmp")
	graph := filepath.Join(dir, "predict_net.pb")
	weights := filepath.Join(dir, "init_net.pb")
	features := filepath.Join(dir, "synset.txt")

	ctx := context.Background()

	defer tracer.Close()

	if _, err := os.Stat(graph); os.IsNotExist(err) {
		if _, err := downloadmanager.DownloadInto(graph_url, dir); err != nil {
			panic(err)
		}
	}
	if _, err := os.Stat(weights); os.IsNotExist(err) {

		if _, err := downloadmanager.DownloadInto(weights_url, dir); err != nil {
			panic(err)
		}
	}
	if _, err := os.Stat(features); os.IsNotExist(err) {

		if _, err := downloadmanager.DownloadInto(features_url, dir); err != nil {
			panic(err)
		}
	}

	imgDir, _ := filepath.Abs("../_fixtures")
	imagePath := filepath.Join(imgDir, "platypus.jpg")

	img, err := imgio.Open(imagePath)
	if err != nil {
		panic(err)
	}

	var input []float32
	for ii := 0; ii < batchSize; ii++ {
		resized := transform.Resize(img, 227, 227, transform.Linear)
		res, err := cvtImageTo1DArray(resized, []float32{123, 117, 104})
		if err != nil {
			panic(err)
		}
		input = append(input, res...)
	}

	opts := options.New()

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE

	} else {
		panic("no GPU")
	}

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.FULL_TRACE, "caffe2_batch")
	defer span.Finish()

	// create predictor
	predictor, err := caffe2.New(
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Graph([]byte(graph)),
		options.Weights([]byte(weights)),
		options.BatchSize(uint32(batchSize)))
	if err != nil {
		panic(err)
	}
	defer predictor.Close()

	/*
		if nvidiasmi.HasGPU {
			cu, err := cupti.New(cupti.Context(ctx))
			if err == nil {
				defer func() {
					cu.Wait()
					cu.Close()
				}()
			}
		}
	*/

	C.cudaProfilerStart()
	predictions, err := predictor.Predict(input, batchSize, 3, 227, 227)
	C.cudaProfilerStop()

	var labels []string
	f, err := os.Open(features)
	if err != nil {
		os.Exit(-1)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}

	len := len(predictions) / batchSize
	for i := 0; i < 1; i++ {
		res := predictions[i*len : (i+1)*len]
		res.Sort()
		pp.Println(res[0].Probability)
		pp.Println(labels[res[0].Index])
	}

	// os.RemoveAll(dir)
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}

package main

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_profiler_api.h>
// #include <cudaProfiler.h>
import "C"

import (
	"bufio"
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
        // TODO: uncomment once blob type error is solved
	//nvidiasmi "github.com/rai-project/nvidia-smi"
)

var (
	batch        = 128
	graph_url    = "http://s3.amazonaws.com/store.carml.org/models/caffe2/squeezenet_1.0/predict_net.pb"
	weights_url  = "http://s3.amazonaws.com/store.carml.org/models/caffe2/squeezenet_1.0/init_net.pb"
	features_url = "http://data.dmlc.ml/mxnet/models/imagenet/synset.txt"
)

// convert go Image to 1-dim array
func cvtImageTo1DArray(src image.Image, mean float32) ([]float32, error) {
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
			res[y*w+x] = float32(b>>8) - mean
			res[w*h+y*w+x] = float32(g>>8) - mean
			res[2*w*h+y*w+x] = float32(r>>8) - mean
		}
	}

	return res, nil
}

func main() {
	defer C.cuProfilerStop()

	dir, _ := filepath.Abs("../tmp")
	graph := filepath.Join(dir, "predict_net.pb")
	weights := filepath.Join(dir, "init_net.pb")
	features := filepath.Join(dir, "synset.txt")

	if _, err := downloadmanager.DownloadInto(graph_url, dir); err != nil {
		os.Exit(-1)
	}

	if _, err := downloadmanager.DownloadInto(weights_url, dir); err != nil {
		os.Exit(-1)
	}

	if _, err := downloadmanager.DownloadInto(features_url, dir); err != nil {
		os.Exit(-1)
	}

	var input []float32
	cnt := 0

	imgDir, _ := filepath.Abs("../_fixtures")
	err := filepath.Walk(imgDir, func(path string, info os.FileInfo, err error) error {
		if path == imgDir || filepath.Ext(path) != ".jpg" || cnt >= batch {
			return nil
		}

		img, err := imgio.Open(path)
		if err != nil {
			return err
		}
		resized := transform.Resize(img, 227, 227, transform.Linear)
		res, err := cvtImageTo1DArray(resized, 128)
		if err != nil {
			panic(err)
		}
		input = append(input, res...)
		cnt++

		return nil
	})
	if err != nil {
		panic(err)
	}

	opts := options.New()

	device := options.CPU_DEVICE
        // TODO: blob type error
        // blob contains caffe2::Tensor<CPUContext>
        // but caller (predictor mostly) expects caffe2:Tensor<CUDAContext>
	/*if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}*/
	pp.Println("Using device = ", device)

	// create predictor
	predictor, err := caffe2.New(
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Graph([]byte(graph)),
		options.Weights([]byte(weights)),
		options.BatchSize(uint32(batch)))
	if err != nil {
		panic(err)
	}
	defer predictor.Close()

	predictions, err := predictor.Predict(input, 10, 3, 227, 227)

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

	len := len(predictions) / batch
	for i := 0; i < cnt; i++ {
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

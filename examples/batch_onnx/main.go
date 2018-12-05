package main

import (
	"bufio"
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"

	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/downloadmanager"
	caffe2 "github.com/rai-project/go-caffe2"
	cupti "github.com/rai-project/go-cupti"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"

	"github.com/rai-project/tracer/ctimer"
	_ "github.com/rai-project/tracer/jaeger"
)

var (
	batchSize   = 1
	model       = "shufflenet"
	weights_url = "https://s3.amazonaws.com/download.onnx/models/opset_8/shufflenet.tar.gz"
	synset_url  = "http://data.dmlc.ml/mxnet/models/imagenet/synset.txt"
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
			res[y*w+x] = (float32(b>>8) - mean[0]) / 255.0
			res[w*h+y*w+x] = (float32(g>>8) - mean[1]) / 225.0
			res[2*w*h+y*w+x] = (float32(r>>8) - mean[2]) / 225.0
		}
	}

	return res, nil
}

func main() {
	defer tracer.Close()

	dir, _ := filepath.Abs("../tmp")
	dir = filepath.Join(dir, model)
	weights := filepath.Join(dir, "model.onnx")
	synset := filepath.Join(dir, "synset.txt")

	if _, err := os.Stat(weights); os.IsNotExist(err) {

		if _, err := downloadmanager.DownloadInto(weights_url, dir); err != nil {
			// note the code may not be able to unzip correctly
			panic(err)
		}
	}
	if _, err := os.Stat(synset); os.IsNotExist(err) {

		if _, err := downloadmanager.DownloadInto(synset_url, dir); err != nil {
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
		resized := transform.Resize(img, 224, 224, transform.Linear)
		res, err := cvtImageTo1DArray(resized, []float32{0, 0, 0})
		if err != nil {
			panic(err)
		}
		input = append(input, res...)
	}

	opts := options.New()

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	ctx := context.Background()

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.FULL_TRACE, "caffe2_batch")
	defer span.Finish()

	predictor, err := caffe2.New(
		ctx,
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Weights([]byte(weights)),
		options.BatchSize(batchSize))
	if err != nil {
		panic(err)
	}
	defer predictor.Close()

	enableCupti := true

	var cu *cupti.CUPTI
	if enableCupti && nvidiasmi.HasGPU {
		cu, err = cupti.New(cupti.Context(ctx))
		if err != nil {
			panic(err)
		}
	}

	predictor.StartProfiling("predict", "")

	err = predictor.Predict(ctx, input, 3, 224, 224)
	if err != nil {
		panic(err)
	}

	predictor.EndProfiling()

	if enableCupti && nvidiasmi.HasGPU {
		cu.Wait()
		cu.Close()
	}

	profBuffer, err := predictor.ReadProfile()
	if err != nil {
		panic(err)
	}
	predictor.DisableProfiling()

	t, err := ctimer.New(profBuffer)
	if err != nil {
		panic(err)
	}
	t.Publish(ctx, tracer.APPLICATION_TRACE)

	output, err := predictor.ReadPredictionOutput(ctx)
	if err != nil {
		panic(err)
	}

	var labels []string
	f, err := os.Open(synset)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}

	features := make([]dlframework.Features, batchSize)
	featuresLen := len(output) / batchSize

	for ii := 0; ii < batchSize; ii++ {
		rprobs := make([]*dlframework.Feature, featuresLen)
		for jj := 0; jj < featuresLen; jj++ {
			rprobs[jj] = feature.New(
				feature.ClassificationIndex(int32(jj)),
				feature.ClassificationLabel(labels[jj]),
				feature.Probability(output[ii*featuresLen+jj]),
			)
		}
		sort.Sort(dlframework.Features(rprobs))
		features[ii] = rprobs
	}

	if true {
		for i := 0; i < 1; i++ {
			results := features[i]
			top1 := results[0]
			pp.Println(top1.Probability)
			pp.Println(top1.GetClassification().GetLabel())
		}
	} else {
		_ = features
	}
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}

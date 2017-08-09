package caffe2

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthonynsimon/bild/parallel"
	"github.com/k0kubun/pp"

	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	homedir "github.com/mitchellh/go-homedir"
	"github.com/stretchr/testify/assert"
)

var (
	homeDir, _         = homedir.Dir()
	initNetlURL        = "https://github.com/caffe2/models/blob/master/squeezenet/exec_net.pb"
	predictNetURL      = "https://github.com/caffe2/models/blob/master/squeezenet/predict_net.pb"
	predictNetFileName = filepath.Join(homeDir, "Downloads", "predict_net.pb")
	initNetFileName    = filepath.Join(homeDir, "Downloads", "exec_net.pb")
	imageFileName      = filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "chicken.jpg")
	// meanImage          = filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "imagenet_mean.binaryproto")
)

func getImageData(t *testing.T, img image.Image) ([]float32, error) {

	b := img.Bounds()
	height := b.Max.Y - b.Min.Y // image height
	width := b.Max.X - b.Min.X  // image width

	res := make([]float32, 3*height*width)
	parallel.Line(height, func(start, end int) {
		w := width
		h := height
		for y := start; y < end; y++ {
			for x := 0; x < width; x++ {
				r, g, b, _ := img.At(x+b.Min.X, y+b.Min.Y).RGBA()
				res[y*w+x] = float32(r>>8) - 128
				res[w*h+y*w+x] = float32(g>>8) - 128
				res[2*w*h+y*w+x] = float32(b>>8) - 128
			}
		}
	})

	return res, nil
}

func TestCreatePredictor(t *testing.T) {
	predictor, err := New(initNetFileName, predictNetFileName)
	assert.NoError(t, err)
	defer predictor.Close()
	assert.NotEmpty(t, predictor)

	imgFile, err := os.Open(imageFileName)
	assert.NoError(t, err)
	defer imgFile.Close()

	image, _, err := image.Decode(imgFile)
	assert.NoError(t, err)
	assert.NotEmpty(t, image)

	// caffenet dim is 227
	imageWidth, imageHeight := image.Bounds().Dx(), image.Bounds().Dy()
	assert.Equal(t, 227, imageWidth)
	assert.Equal(t, 227, imageHeight)

	imageData, err := getImageData(t, image)
	assert.NoError(t, err)

	predictions, err := predictor.Predict(imageData)
	predictions.Sort()
	pp.Println(predictions[0:2])

	assert.NoError(t, err)
	assert.NotEmpty(t, predictions)
	assert.Equal(t, 1000, len(predictions))
}

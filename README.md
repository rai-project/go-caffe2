# Go Bindings for Caffe2 [![Build Status](https://travis-ci.org/rai-project/go-caffe2.svg?branch=master)](https://travis-ci.org/rai-project/go-caffe2) [![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/go-caffe2)](https://goreportcard.com/report/github.com/rai-project/go-caffe2)[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)[![](https://images.microbadger.com/badges/version/carml/go-caffe2:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:ppc64le-gpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe2:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe2:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe2:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:amd64-gpu-latest 'Get your own version badge on microbadger.com')

## Caffe2 Installation

Please refer to [scripts](scripts) or the `LIBRARY INSTALLATION` section in the [dockefiles](dockerfiles) to install Caffe2 on your system. OpenBLAS is used.

If you get an error about not being able to write to `/opt` then perform the following

```
sudo mkdir -p /opt/pytorch
sudo chown -R `whoami` /opt/pytorch
```

- The default blas is OpenBLAS.
  The default OpenBLAS path for mac os is `/usr/local/opt/openblas` if installed throught homebrew (openblas is keg-only, which means it was not symlinked into /usr/local, because macOS provides BLAS and LAPACK in the Accelerate framework).

- The default Caffe2 installation path is `/opt/pytorch/caffe2`.

- The default CUDA path is `/usr/local/cuda`

See [lib.go](lib.go) for details.

After installing Caffe2, run `export DYLD_LIBRARY_PATH=/opt/pytorch/caffe2/lib:$DYLD_LIBRARY_PATH` on mac os or `export LD_LIBRARY_PATH=/opt/pytorch/caffe2/lib:$DYLD_LIBRARY_PATH`on linux.

## Use Other Libary Paths

To use different library paths, change CGO_CFLAGS, CGO_CXXFLAGS and CGO_LDFLAGS enviroment variables.

For example,

```
    export CGO_CFLAGS="${CGO_CFLAGS} -I /usr/local/cuda-9.2/include -I/usr/local/cuda-9.2/nvvm/include -I /usr/local/cuda-9.2/extras/CUPTI/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include/crt"
    export CGO_CXXFLAGS="${CGO_CXXFLAGS} -I /usr/local/cuda-9.2/include -I/usr/local/cuda-9.2/nvvm/include -I /usr/local/cuda-9.2/extras/CUPTI/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include/crt"
    export CGO_LDFLAGS="${CGO_LDFLAGS} -L /usr/local/nvidia/lib64 -L /usr/local/cuda-9.2/nvvm/lib64 -L /usr/local/cuda-9.2/lib64 -L /usr/local/cuda-9.2/lib64/stubs -L /usr/local/cuda-9.2/targets/x86_64-linux/lib/stubs/ -L /usr/local/cuda-9.2/lib64/stubs -L /usr/local/cuda-9.2/extras/CUPTI/lib64"
```

Run `go build` in to check the Caffe2 installation and library paths set-up.

## Run Then Examples

Make sure you have already [install mlmodelscope dependences](https://docs.mlmodelscope.org/installation/source/dependencies/) and [set up the external services](https://docs.mlmodelscope.org/installation/source/external_services/).

On linux, the default is to use GPU, if you don't have a GPU, do `go build -tags nogpu` instead of `go build`.

### batch

This example is to show how to use mlmodelscope tracing to profile the inference.

```
  cd example/batch
  go build
  ./batch
```

Then you can go to `localhost:16686` to look at the trace of that inference.

### batch_nvprof

You need GPU and CUDA to run this example. This example is to show how to use nvprof to profile the inference.

```
  cd example/batch_nvprof
  go build
  nvprof --profile-from-start off ./batch_nvprof
```

Refer to [Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html) on how to use nvprof.

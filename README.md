# Go Bindings for Caffe2 Prediction [![Build Status](https://travis-ci.org/rai-project/go-caffe2.svg?branch=master)](https://travis-ci.org/rai-project/go-caffe2) [![](https://images.microbadger.com/badges/version/carml/go-caffe2:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:ppc64le-gpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe2:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe2:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe2:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:amd64-gpu-latest 'Get your own version badge on microbadger.com')

## Caffe2 Installation

Please refer to the `scripts/build_caffe2.sh` or `LIBRARY INSTALLATION` section in the [dockefiles](dockerfiles) to install caffe2 on your system.

- The default blas is OpenBLAS.

{{% notice note %}}
The default OpenBLAS path for mac os is `/usr/local/opt/openblas` if installed throught homebrew (openblas is keg-only, which means it was not symlinked into /usr/local, because macOS provides BLAS and LAPACK in the Accelerate framework).
{{% /notice %}}

- The default caffe2 installation path is `/opt/caffe2` for linux, darwin and ppc64le w/o powerai; `/opt/DL/caffe2` for ppc64le w/ powerai.

* The default CUDA path is `/usr/local/cuda`

See [lib.go](lib.go) for details.

## Use Other Libary Paths

To use other library paths, change CGO_CFLAGS, CGO_CXXFLAGS and CGO_LDFLAGS enviroment variables.

For example,

```
    export CGO_CFLAGS="${CGO_CFLAGS} -I /usr/local/cuda-9.2/include -I/usr/local/cuda-9.2/nvvm/include -I /usr/local/cuda-9.2/extras/CUPTI/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include/crt"
    export CGO_CXXFLAGS="${CGO_CXXFLAGS} -I /usr/local/cuda-9.2/include -I/usr/local/cuda-9.2/nvvm/include -I /usr/local/cuda-9.2/extras/CUPTI/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include/crt"
    export CGO_LDFLAGS="${CGO_LDFLAGS} -L /usr/local/nvidia/lib64 -L /usr/local/cuda-9.2/nvvm/lib64 -L /usr/local/cuda-9.2/lib64 -L /usr/local/cuda-9.2/lib64/stubs -L /usr/local/cuda-9.2/targets/x86_64-linux/lib/stubs/ -L /usr/local/cuda-9.2/lib64/stubs -L /usr/local/cuda-9.2/extras/CUPTI/lib64"
```

## Run the examples

### batch

### batch_nvprof

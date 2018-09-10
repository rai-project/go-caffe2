# GO Bindings for Caffe2 Prediction [![Build Status](https://travis-ci.org/rai-project/go-caffe2.svg?branch=master)](https://travis-ci.org/rai-project/go-caffe2) [![](https://images.microbadger.com/badges/version/carml/go-caffe2:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:ppc64le-gpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe2:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe2:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe2:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe2:amd64-gpu-latest 'Get your own version badge on microbadger.com')

## Caffe2 Installation

The go bindings need a Caffe2 installation and to include caffe2 path in [lib.go](lib.go).

Please refer to the `LIBRARY INSTALLATION` section in the [dockefiles](dockerfiles) to install caffe on your system.

The default caffe installation path is `/opt/caffe` for linux, darwin and ppc64le. See [lib.go](lib.go) for details.

## CUDA Installation

The default CUDA path is `/usr/local/cuda`

#!/bin/bash

cd 3rdparty/tilelang

if [ "$1" == "--update" ] || [ "$1" == "-u" ]; then
    echo "Running incremental build via cmake..."
    cmake --build build -- -j
    exit 0
fi

pip install -r requirements.txt
pip install apache-tvm-ffi==0.1.9
pip install cmake==3.26.1
pip install ninja cython scikit-build-core

pip install -e . -v --no-build-isolation

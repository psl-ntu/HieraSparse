#!/bin/bash

if ! command -v conda &> /dev/null; then
    echo "conda is not installed."
    exit 1
fi

if ! conda info --envs | grep -q "^hierasparse[[:space:]]"; then
    conda create -y -n hierasparse python=3.10
fi

eval "$(conda shell.bash hook)"
conda activate hierasparse

pip install -e . -v --no-build-isolation 2>&1 | tee hierasparse_install.log

if ! python -c "import flash_attn" &> /dev/null; then
    pip install flash-attn -v --no-build-isolation 2>&1 | tee fa2_install.log
fi

if ! python -c "import tilelang" &> /dev/null; then
    bash "$(dirname "$0")/install_tilelang.sh" 2>&1 | tee tilelang_install.log
fi

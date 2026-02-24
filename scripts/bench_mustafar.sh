#!/bin/bash

python benchmark/mustafar_baseline.py --key_sparsity 0.5 --value_sparsity 0.5
python benchmark/mustafar_baseline.py --key_sparsity 0.5 --value_sparsity 0.0
python benchmark/mustafar_baseline.py --key_sparsity 0.0 --value_sparsity 0.5

#!/bin/bash

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
echo "Job Started"
python test.py ../pretrained_models ucsd_maskopt_exp_3 1024 ../data/x_test_inl.npy ../data/x_test_out.npy

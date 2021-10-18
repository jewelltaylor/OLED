#!/bin/bash

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
echo "Job Started"
python test.py ../pretrained_models cifar_maskopt_exp_fin_7 ../data/x_test.npy ../data/y_test.npy

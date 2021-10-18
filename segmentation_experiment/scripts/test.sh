#!/bin/bash

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
echo "Job Started"
python test.py ../pretrained_models/mask_model_auc_context.h5 ../data/x_train.npy ../data/y_train.npy ../data/x_test.npy ../data/y_test.npy

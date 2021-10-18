#!/bin/bash

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
echo "Job Started"
python train.py train_run ../models 400 1024 ../data/x_train.npy ../data/x_val_inl.npy ../data/x_val_out.npy

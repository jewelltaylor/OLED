#!/bin/bash

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
echo "Job Started"
python test_trained.py ../models train_run ../data/x_train.npy ../data/y_train.npy ../data/x_test.npy ../data/y_test.npy

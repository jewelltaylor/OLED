#!/bin/bash

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python train.py train_run_0 ../models 0 400 256 ../data/x_train.npy ../data/y_train.npy
python train.py train_run_1 ../models 1 400 256 ../data/x_train.npy ../data/y_train.npy
python train.py train_run_2 ../models 2 400 256 ../data/x_train.npy ../data/y_train.npy
python train.py train_run_3 ../models 3 400 256 ../data/x_train.npy ../data/y_train.npy
python train.py train_run_4 ../models 4 400 256 ../data/x_train.npy ../data/y_train.npy
python train.py train_run_5 ../models 5 400 256 ../data/x_train.npy ../data/y_train.npy
python train.py train_run_6 ../models 6 400 256 ../data/x_train.npy ../data/y_train.npy
python train.py train_run_7 ../models 7 400 256 ../data/x_train.npy ../data/y_train.npy
python train.py train_run_8 ../models 8 400 256 ../data/x_train.npy ../data/y_train.npy
python train.py train_run_9 ../models 9 400 256 ../data/x_train.npy ../data/y_train.npy

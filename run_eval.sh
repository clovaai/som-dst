#!/bin/sh

MODEL_PATH='./model_2.1/model_best.bin'
DATASET_DIR='./data/mw_2.1/'

python evaluation.py\
    --model_ckpt_path $MODEL_PATH\
    --data_root $DATASET_DIR\

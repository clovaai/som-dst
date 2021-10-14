DATASET_DIR='./data/mw_2.1/'
SAVE_DIR='./model_2.1/'

python train.py\
    --data_root ${DATASET_DIR}\
    --save_dir ${SAVE_DIR}\
    --bert_ckpt_path 'bert-base-uncased-pytorch_model.bin'\
    --op_code '4'
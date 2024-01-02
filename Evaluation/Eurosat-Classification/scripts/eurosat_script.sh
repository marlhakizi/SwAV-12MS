DUMP_PATH="./dr/"
python Eurosat_eval/eurosat_train.py\
    --log_level "INFO"\
    --dump_path $DUMP_PATH\
    --encoder "swav"\
    --lr 1e-3\
    --normalization "data"\
    --weight_decay 0.0\
    --epochs 5\
    --batch_size 16\
    --device "cuda:4"\
    --data_size 500\
    --name "Eurosat"\
    --nth 1
          
#!/bin/bash
​
DUMP_PATH="./results/field_delineation/"
​​#This is for when we load a Resnet50 model with Imagenet pretrained weights.
#For normalization we use imagenet weights
for size in {64,128,256,512}
do
    for time in {0,1,2,3,4} 
    do
    python main.py\
        --log_level "INFO"\
        --dump_path $DUMP_PATH\
        --task "crop_delineation"\
        --encoder "swav-12"\
        --decoder 'unet'\
        --fine_tune_encoder False\
        --lr 1e-3\
        --normalization "dataTwelve"\
        --weight_decay 0.0\
        --epochs 100\
        --batch_size 16\
        --device "cpu"\
        --criterion "softiou"\
        --data_size $size\
        --name "Swav_12"\
        --nth $time
    done
done               
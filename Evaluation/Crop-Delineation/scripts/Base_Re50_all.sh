#!/bin/bash
​
DUMP_PATH="./results/field_delineation/"
​#change data size and type depending on need
​#This is for when we load a basic Resnet50 model with no pretrained weights.
#For normalization we use crop-del data
for size in {1095,64,128,256,512}
do
    for time in {1,2,3,4} 
    do
    python main.py\
        --log_level "INFO"\
        --dump_path $DUMP_PATH\
        --task "crop_delineation"\
        --encoder 'none'\
        --decoder 'unet'\
        --fine_tune_encoder False\
        --lr 1e-3\
        --normalization "dataTwelve"\
        --weight_decay 0.0\
        --epochs 100\
        --batch_size 16\
        --device "cuda:0"\
        --criterion "softiou"\
        --data_size $size\
        --name "Base_Res50_Twelve"\
        --nth $time
    done
done       
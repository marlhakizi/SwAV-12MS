#!/bin/bash
​
DUMP_PATH="./results/field_delineation/"
​​#This is for when we load a Resnet50 model with Imagenet pretrained weights.
#For normalization we use imagenet weights
cd Evaluation
for size in {1095,64,128,256,512}
do
    for time in {0,1,2,3,4} 
    do
    python main.py\
        --log_level "INFO"\
        --dump_path $DUMP_PATH\
        --task "crop_delineation"\
        --encoder 'imagenet'\
        --decoder 'unet'\
        --fine_tune_encoder False\
        --lr 1e-3\
        --normalization "imagenet"\
        --weight_decay 0.0\
        --epochs 1\
        --batch_size 20\
        --device "cpu"\
        --criterion "softiou"\
        --data_size $size\
        --name "Base_Res50_RGB"\
        --nth $time
    done
done                 
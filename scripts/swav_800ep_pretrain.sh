#!/bin/bash
TWELVE_DATASET_PATH="/scratch/mh613/tenk_twelve/"
TEN_DATASET_PATH="/scratch/mh613/hundredk/"
RGB_DATASET_PATH="/scratch/mh613/new_10K_RGB/"
DUMPPATH="./anotherpath/"
mkdir -p $DUMPPATH
torchrun --standalone --nnodes=1 --nproc_per_node=8 main_swav.py \
--data_path $TWELVE_DATASET_PATH \
--nmb_crops 2 6 \
--initialize_imagenet False \
--number_channels 12 \
--size_crops 160 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.03 \
--feat_dim 128 \
--nmb_prototypes 500 \
--queue_length 0 \
--epochs 200 \
--freeze_prototypes_niters 3000 \
--batch_size 32 \
--base_lr 4.8 \
--sync_bn apex \
--final_lr 0.0048 \
--wd 0.000001 \
--warmup_epochs 10 \
--arch resnet50 \
--use_fp16 true \
--task 12_channels \
--dump_path $DUMPPATH
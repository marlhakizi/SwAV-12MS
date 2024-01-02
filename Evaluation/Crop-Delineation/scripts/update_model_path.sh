#!/bin/bash

MODEL_PATH="/home/mh613/updatedswav/millionpaath/checkpoints/ckp-eval.pth"
OUTPUT_PATH="/home/mh613/updatedswav/millionpaath/checkpoints/ckp-eval.pth"
python save_model.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH 
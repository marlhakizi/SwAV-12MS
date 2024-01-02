# ClimateEye

ClimateEye provides a tool where the user can:

- Download satellite images with 12 channel bands 
- Visualize different band combinations
- Run a self-supervised model - SwAV - with no labels needed. 
- Evaluate the above pre-trained model on sample data

This repository provides necessary instructions to:
- Setup your virtual environment
- Download 12-channel images
- Run different versions of the SwAV model (on RGB or 12-channel data).
- Run SwAV on different downstream, climate-related tasks

## Environment Setup

To get started, install a new virtual environment with necessary packages:
```python
conda env create -f environment.yml
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- See Environment_Setup/Environment setup.pdf document under "EXISTING ISSUES" for issues that might arise and how to fix them.

## Download Images

Before downloading images, a few steps are needed beforehand:

- A personal or service account to authenticate to Google Earth Engine, as well as a private JSON for that account. Here is a [link](https://developers.google.com/earth-engine/guides/python_install#authentication) with instructions on authentication using the Python API.
- A csv files with coordinates info. See Data/coordinates.csv for a sample.

After editing download.py for your local machin, run:

```python
python data_download.py
```
This downloads 12-channel data as TIFF files.

## Visualize

- The notebook found in Visualize/visualize.ipynb goes through how to plot different band combinations from a tiff file. 

# SwAV

<div align="center">
  <img width="100%" alt="SwAV Illustration" src="https://dl.fbaipublicfiles.com/deepcluster/animated.gif">
</div>

SwAV is an efficient and simple method for pre-training convnets without using annotations.
Similarly to contrastive approaches, SwAV learns representations by comparing transformations of an image, but unlike contrastive methods, it does not require to compute feature pairwise comparisons.
It makes our framework more efficient since it does not require a large memory bank or an auxiliary momentum network. The method simultaneously clusters the data while enforcing consistency between cluster assignments produced for different augmentations (or “views”) of the same image, instead of comparing features directly. It uses a “swapped” prediction mechanism where it predicts the cluster assignment of a view from the representation of another view.

## SwAV training

To run the model on RGB or 12-channel data. Run 

```python
./scripts/swav_800ep_pretrain.sh
```
where data_path, number_channels (3 or 12) and task ("RGB" or "12-channels") will need to change depending on whether the user wants to train  the model on RGB or 12-channel.

For RGB, it would look like:

```
RGB_DATASET_PATH="/RGB_path"
torchrun main_swav.py \
--data_path $RGB_DATASET_PATH \ 
--nmb_crops 2 6 \
--number_channels 3 \
--size_crops 160 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--feat_dim 128 \
--nmb_prototypes 100 \
--queue_length 0 \
--epochs 800 \
--batch_size 32 \
--base_lr 0.5 \
--final_lr 0.0005 \
--wd 0.000001 \
--warmup_epochs 0 \
--freeze_prototypes_niters 5005 \
--arch resnet50 \
--use_fp16 true \
--task RGB
```

and 12-channels would be :

```
RGB_DATASET_PATH="/RGB_path"
torchrun main_swav.py \
--data_path $TWELVE_DATASET_PATH \ 
--nmb_crops 2 6 \
--number_channels 12 \
--size_crops 160 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--feat_dim 128 \
--nmb_prototypes 100 \
--queue_length 0 \
--epochs 800 \
--batch_size 32 \
--base_lr 0.5 \
--final_lr 0.0005 \
--wd 0.000001 \
--warmup_epochs 0 \
--freeze_prototypes_niters 5005 \
--arch resnet50 \
--use_fp16 true \
--task 12-channels
```

## Key Findings

Preliminary results show that using 12-channel images for model pre-training does increase model performance relative to using RGB images for model-pretraining. More details about the results of our experiments can be found in Documents/Final_writeup.

We use the IoU metric (intersection over union) to measure performance on a crop delineation task. IoU is a common metric used in computer vision applications. Pre-trained on RGB data, the SwAV model achieved an IoU score of 0.463, and when pre-trained on 12-channel data, it achieved an IoU score of 0.52, which is a 12% increase in performance.

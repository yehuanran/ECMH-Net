# ECMH-Net: Edge-Guided ConvNeXt-Mamba Hybrid Network for High-Resolution Remote Sensing Building Extraction

This is an official PyTorch implementation of "[**ECMH-Net: Edge-Guided ConvNeXt-Mamba Hybrid Network for High-Resolution Remote Sensing Building Extraction**]".

# Introduction
Extracting buildings from high-resolution remote sensing imagery is essential for urban planning and disaster management. However, complex geospatial backgrounds, extreme scale variations, and severe shadow occlusions all exacerbate the difficulty of accurate boundary delineation. Convolutional neural networks excel at capturing high-frequency local details but fall short in establishing long-range spatial dependencies. Although vision transformers and state space models can provide global contextual reasoning capabilities, flattening a two-dimensional feature grid into a one-dimensional sequence fundamentally disrupts the geospatial topological structure, resulting in sawtoothlike anomalies at building corners. To overcome these limitations,we propose the Edge-Guided ConvNeXt-Mamba Hybrid Network (ECMH-Net).
<center> 
<img src="DRAU-Net.png" width="auto" height="auto">
</center>

# Image segmentation

## 1. Requirements
```
# Environments:
cuda==12.6
python==3.10
# Dependencies:
pip install torch==2.7.1 torchvision==0.22.1
pip install einops==0.6.1 imageio==2.28.1   albumentations   Torchmetrics==0.11.4
```

## 2. Data Preparation

```
│inria/
├──austin1/
│  ├── images
│  │   ├── austin1.jpg
│  │   ├── ......
│  ├── binary_masks
├──austin2/
│  ├── images
│  │   ├── austin2.jpg
│  │   ├── ......
│  ├── ......
```

## 3. Train

python train.py --dataset inria 

## 4. Validation

python test.py --load_checkpoint output/checkpoints/20250401-0413_unet/20250401-0413_unet_e1.pt


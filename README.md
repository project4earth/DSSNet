# DSSNet: Dual Stream Segmentation Networks

<p align="center">
  <img src="./doc/dssnet_architecture.png" alt="DSSNet Architecture" width="60%">
</p>

## Overview
**DSSNet** is a lightweight dual-encoder semantic segmentation framework that fuses Sentinel-1 SAR and Sentinel-2 optical imagery. DSSNet leverages modality-specific backbones from different architectural paradigms: EfficientNet-B0, a convolutional, and MaxVit-T, a transformer-based encoder.  

The full publication describing DSSNet is available at:  
**[https://doi.org/10.1016/j.rsase.2026.101895](https://doi.org/10.1016/j.rsase.2026.101895)**

## Pretrained Weights
We provide pretrained weights for DSSNet and baseline models:
- [Pretrained Models](https://drive.google.com/drive/folders/1VRIxd15jHAUe421hNSNV7YKZXpXhvUQi?usp=sharing)  

## Getting Started
### Requirements
- Python 3.11+
- PyTorch >= 2.3.1
- Torchvision >= 0.20

### Installation
Clone the repository:
```bash
git clone https://github.com/project4earth/DSSNet.git
cd DSSNet
```

### Citation
If you use DSSNet in your research, please cite:
```bash
@article{WIJAYA2026101895,
title = {Lightweight dual-encoder deep learning integrating Sentinel-1 and Sentinel-2 for paddy field mapping},
journal = {Remote Sensing Applications: Society and Environment},
pages = {101895},
year = {2026},
issn = {2352-9385},
doi = {https://doi.org/10.1016/j.rsase.2026.101895},
url = {https://www.sciencedirect.com/science/article/pii/S2352938526000285},
author = {Bagus Setyawan Wijaya and Rinaldi Munir and Nugraha Priya Utama}
}
```

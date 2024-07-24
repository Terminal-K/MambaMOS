<p align="center">
<h1 align="center"><strong>MambaMOS: LiDAR-based 3D Moving Object Segmentation with Motion-aware State Space Model</strong></h1>
<h3 align="center">ACM MM 2024</h3>

<div align="center">
<div>
    <a href="https://arxiv.org/abs/2404.12794"><img src="http://img.shields.io/badge/paper-arXiv.cs.CV%3A2404.12794-B31B1B.svg"></a>
  </div>
</div>

<p align="center">
<img src="./assets/overview.png" width="95%">
</p>
<b><p align="center" style="margin-top: -20px;">
MambaMOS
</b></p>

## Overview
- [Environment](#environment)
- [Dataset preparation](#dataset-preparation)
- [Run](#run)
- [Acknowledgement](#acknowledgement)

## Environment
```
# pointcept with CUDA=11.6
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

pip install torch-geometric
pip install spconv-cu116
pip install open3d

cd libs/pointops
python setup.py install
cd ../../

# mamba install
cd libs/
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.1.3 
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ..
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v1.1.4 
MAMBA_FORCE_BUILD=TRUE pip install .
```

## Dataset preparation
- Download [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) dataset.
- Link dataset to codebase.
```bash
mkdir -p data
ln -s ${SEMANTIC_KITTI_DIR} ${CODEBASE_DIR}/data/semantic_kitti
```
Data structure:
```
SEMANTIC_KITTI_DIR
└── sequences
    ├── 00
    │   ├── velodyne
    │   │    ├── 000000.bin
    │   │    ├── 000001.bin
    │   │    └── ...
    │   ├── labels
    │   │    ├── 000000.label
    │   │    ├── 000001.label
    │   │    └── ...
    │   ├── calib.txt
    │   ├── poses.txt
    │   └── times.txt
    ├── 01
    ├── 02
   ...
    └── 21

# sequences for training: 00-07, 09-10
# sequences for validation: 08
# sequences for testing: 11-21
```

## Run
### Training
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
sh scripts/train.sh -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
```
For example:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
sh scripts/train.sh -g 4 -d semantic_kitti -c semseg_mambamos -n demo
```

### Testing
In the testing phase, we used the same testing strategy as [pointcept](https://github.com/Pointcept/Pointcept), please read [its readme](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#testing) for information.
```bash
# By script (Based on experiment folder created by training script)
sh scripts/test.sh -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
```
For example:
```bash
export CUDA_VISIBLE_DEVICES=0
# weight path: ./exp/semantic_kitti/mambamos/model_best.pth
sh scripts/test.sh -g 1 -d semantic_kitti -n mambamos -w model_best
```
Our pretrained model is public available and can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1oZ39MqgKx9kpBKRZW5-9Ui8oo_8KvSe4?usp=drive_link).

## 🤝 Publication:
Please consider referencing this paper if you use the ```code``` from our work.
Thanks a lot :)

```
@inproceedings{zeng2024mambamos,
  title={MambaMOS: LiDAR-based 3D Moving Object Segmentation with Motion-aware State Space Model},
  author={Zeng, Kang and Shi, Hao and Lin, Jiacheng and Li, Siyu and Cheng, Jintao and Wang, Kaiwei and Li, Zhiyong and Yang, Kailun},
  booktitle={ACM International Conference on Multimedia (MM)},
  year={2024}
}
```


## Acknowledgement
The code framework of this project is based on [pointcept](https://github.com/Pointcept/Pointcept), 
and the code of _MambaMOS_ and _MSSM_ refers to [PTv3](https://github.com/Pointcept/PointTransformerV3) and [mamba](https://github.com/state-spaces/mamba) respectively, thanks to their excellent work.

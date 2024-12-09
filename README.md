# SILM

## Get Started!

### 1. Prepare Conda Environment
CUDA Version: 11.3

Python Version: 3.8 

You can install different version of PyTorch according to your CUDA version.

```bash
conda create -n silm python=3.8
conda activate silm
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 --extra-index-url https://download.pytorch.org/whl/cu111
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple torch_geometric==2.5.0
pip install pytorch-lightning==1.5.2 
```

### 2. Download Code and Dataset
Download the code from the repository:
```bash
git clone https://github.com/Elaine-Blue/SILM.git
cd SILM
```
You can download the Argoverse2 Sensor and Lidar dataset from [Argoverse2 Webpage](https://www.argoverse.org/av2.html). 

To download the full dataset(~1TB), you can download the dataset using the following command:
```bash
cd Demo
bash download_av2_data.sh
```
For simply evaluating the model, you can download the mini-versiobn raw and processed dataset from [Google Drive](https://drive.google.com/drive/folders/1vfIjnIX83S5WqlN25ptcSAP1lYo_Hjyt?usp=sharing) and unzip it to the `SILM/data/av2` folder.
Please make sure the structure of SILM is as follows:
```
SILM/
├── ckpts/
│   ├── bevformer_r101_dcn_24ep.pth
│   ├── movenet.pth
├── data/
│   ├── av2/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   ├── processed/
│   │   ├── trainval/
│   │   |   |── av2_infos_train.pkl
│   │   |   |── av2_infos_val.pkl
├── dataset/
├── mmdet3d/
├── models/
├── tools/
├── train.py
├── val.py
```
### 3. Train Model
```bash
python train.py --root data/av2 --embed_dim 64
```

### 4. Evaluate Model
```bash
python val.py --root data/av2 --ckpt checkpoints/xx.pth
```

### 5. Visualization
Remained to be updated!
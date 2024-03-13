# Investigating Chiral Morphogenesis of Gold using Generative Cellular Automata

This repository contains the code for our Nature Materials 2024 paper: Investigating Chiral Morphogenesis of Gold using Generative Cellular Automata.
The code provides scripts for training Generative Cellular Automata (GCA) and inferencing on various seeds.


## Installation & Data Preparation

### Anaconda environment installation guide
```
conda create -n gca python=3.8
conda activate gca

# install torch
# for cuda 11.2
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
# for cuda 10.2
pip install torch==1.7.1 torchvision==0.8.2

# install MinkowksiEngine
conda install openblas-devel -c anaconda 
export CXX=g++-7
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
# If --install-option does not work (might be deprecated), use the following:
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --config-settings="--blas_include_dirs=${CONDA_PREFIX}/include" --config-settings="--blas=openblas"

# install all other requirements
pip install -r requirements.txt

 install torch-scatter
# for cuda 11.2
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
# for cuda 10.2
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
# for other cuda versions you can find installation guide in https://github.com/rusty1s/pytorch_scatter
```


### Data Preparation
Download the data folder from the [link](https://drive.google.com/drive/folders/13TeuZLReefBHeTUTiRo6r70jr38fjBfr?usp=sharing) containing the point cloud of the morphologies and the pretrained models.
Place it as the below, where the root of the code directory should look like the following:
```
- data/
    - processed/
        - train1_1.ply
        - train1_2.ply
        - train1_3.ply
        - ...
- logs/ 
    - h3/    # if you want inference on pretrained models
        - config.yaml
        - ckpts/
            - ckpt-step-5000
    - ...
- configs/
- models/
- ...
    
```


## Training and inference on GCA
This will produce the ply sequences for each growth reported in the paper every 1000 training steps in the `./log` directory.
The log directory will contain the visualizations in ply format of how each state should look like.
Because the state refers the growth of each state, you'll need to accumulate up to current state to reproduce the visualizations made in the paper.
* Training with hellicoid3:
`export OMP_NUM_THREADS=24; python main.py --config configs/h3.yaml --override "" --log-dir log/h3`
* Training with hellicoid1:
`export OMP_NUM_THREADS=24; python main.py --config configs/h1.yaml --override "" --log-dir log/h1`

## Inference with pretrained models
Running pretrained models
`export OMP_NUM_THREADS=24; python main.py --vis --config configs/h3 --resume-ckpt logs/h3/ckpts/ckpt-step-5000 --override "vis.imgs.step=1" --log-dir logs/h3`


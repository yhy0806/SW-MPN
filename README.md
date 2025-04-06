# SW-MPN
**Abstract**—Open set recognition (OSR) aims to identify unknown classes and distinguish known classes simultaneously. Existing prototype-based OSR methods still risk misclassifying unknown samples as known classes. Transformer-based OSR approaches, primarily using Vision Transformer (ViT), have been limited to visual tasks and struggle to handle intra-class diversity, making them unsuitable for audio datasets. To overcome these limitations, we propose a novel framework called multi-prototype network with Swin Transformer (SW-MPN). The network is developed based on a novel multi-prototype learning mechanism that combines Euclidean distance and dot product similarity to measure the relationship between samples and prototypes. Furthermore, SW-MPN employs the Swin Transformer as feature extractor and replaces its original classification head with a multi-prototype classifier. Our extensive evaluation on both visual and audio tasks shows that our approach significantly outperforms other baseline methods and obtains new state-of-the-art performance for OSR.

![论文2图1](https://github.com/user-attachments/assets/70f3cf72-9ea5-421a-b5a4-35d408d4979a)

# Requirements
First, install python 3.8 or higher. Then:
```shell
# create an environment
conda create -n OSR python=3.8
conda activate OSR

# Install following packages
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn==0.24.2 
```
# Training & Evaluation
To train open set recognition models on image dataset in paper, run this command:
On image dataset

```train
python osr.py --dataset <DATASET> --gpu <GPU> --lr <learning rate> --dataroot <Dataroot>
```
On audio dataset
```train
python Audio_osr.py --dataset <DATASET> --gpu <GPU> --lr <learning rate> --dataroot <Dataroot>
```

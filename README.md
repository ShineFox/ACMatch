# ACMatch
![Framework]('https://github.com/ShineFox/ACMatch/blob/main/framework.png')
The official implementation of the ISPRS P&amp;RS paper "ACMatch: Improving context capture for two-view correspondence learning via adaptive convolution".

## Demo
To start up with our repository, you can first use the demo with the downloaded [pretrained model](https://drive.google.com/drive/folders/18TIQ3E_Vj95tF8u7wQECkTxX0wWjS6NB?usp=drive_link).  
  `CUDA_VISIBLE_DEVICES=0 python demo.py`

## Testing
For relative pose estimation, you can download our pre-trained models [here](https://drive.google.com/drive/folders/18TIQ3E_Vj95tF8u7wQECkTxX0wWjS6NB?usp=drive_link), and then use the following command for testing:  
  `CUDA_VISIBLE_DEVICES=0 python test.py`

## Training
For training the model with YFCC100M, you need to download the training datasets as same as [OANet](https://github.com/zjhthu/OANet), and next you can use the following command to train your own model:  
  `CUDA_VISIBLE_DEVICES=0 python train.py`


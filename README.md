# <p align="center">ACMatch: Improving context capture for two-view correspondence learning via adaptive convolution</p>

<div align="center">
  <a href="https://scholar.google.com/citations?user=tU_vwPwAAAAJ&hl=zh-CN">Xiang Fang</a><sup>a</sup>, 
  <a href="https://scholar.google.com/citations?user=h-9Ub_cAAAAJ&hl=zh-CN&oi=ao">Yifan Lu</a><sup>a</sup>, 
  <a href="https://scholar.google.com/citations?hl=zh-CN&user=7f_tYK4AAAAJ">Shihua Zhang</a><sup>a</sup>, 
  Yining Xie</a><sup>b</sup>, 
  <a href="https://scholar.google.com/citations?user=73trMQkAAAAJ&hl=zh-CN&oi=ao">Jiayi Ma</a><sup>a</sup>
  <p><sup>a</sup>Wuhan University,  <sup>b</sup>Northeast Forestry University</p>
</div>

![Framework](https://github.com/ShineFox/ACMatch/blob/main/framework.png)  
The official implementation of the ISPRS P&amp;RS paper "[ACMatch: Improving context capture for two-view correspondence learning via adaptive convolution](https://www.sciencedirect.com/science/article/pii/S092427162400412X)".

## Environment
Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.7.0). Other dependencies should be easily installed through pip or conda.

## Demo
To start up with our repository, you can first use the demo with the downloaded [pretrained model](https://drive.google.com/drive/folders/18TIQ3E_Vj95tF8u7wQECkTxX0wWjS6NB?usp=drive_link).  
    
    cd ./demo
    CUDA_VISIBLE_DEVICES=0 python demo.py

## Testing
For relative pose estimation, you can download our pre-trained models [here](https://drive.google.com/drive/folders/18TIQ3E_Vj95tF8u7wQECkTxX0wWjS6NB?usp=drive_link), and then use the following command for testing:  

    cd ./test
    CUDA_VISIBLE_DEVICES=0 python test.py

## Training
For training the model with YFCC100M, you need to download the training datasets as same as [OANet](https://github.com/zjhthu/OANet), and next you can use the following command to train your own model:  

    cd ./core
    CUDA_VISIBLE_DEVICES=0 python train.py
### Visualization
![visualization](https://github.com/ShineFox/ACMatch/blob/main/visualization.png)  

## Acknowledgement
We have largely studied and borrowed from the following papers or code, and are very grateful for the inspiration these works have caused us:   
(1) OANet:  ICCV'19, [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_Two-View_Correspondences_and_Geometry_Using_Order-Aware_Network_ICCV_2019_paper.pdf), [code](https://github.com/zjhthu/OANet);  
(2) ConvMatch: AAAI'23, [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25456), [code](https://github.com/SuhZhang/ConvMatch).

## Citation
If you find our work useful, plz cite  

    @article{fang2024acmatch,
    title={ACMatch: Improving context capture for two-view correspondence learning via adaptive convolution},
    author={Fang, Xiang and Lu, Yifan and Zhang, Shihua and Xie, Yining and Ma, Jiayi},
    journal={ISPRS Journal of Photogrammetry and Remote Sensing},
    volume={218},
    pages={466--480},
    year={2024},
    publisher={Elsevier}
    }

If you have ant question, feel free to contact me via [email](xiangfang@whu.edu.cn)!


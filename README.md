# Precise Forecasting of Sky Images Using Spatial Warping

<h4 align="center"><b><a href="https://leronjulian.github.io/" target="_blank">Leron
Julian</a>, <a href="https://www.ece.cmu.edu/directory/bios/sankaranarayanan-aswin.html" target="_blank">Aswin
Sankaranarayanan</a></b></h4>

<h6 align="center"><i>Image Science Lab, Carnegie Mellon University</i></h6>

<h4 align="center">
<a href="http://imagesci.ece.cmu.edu/files/paper/2021/SkyNet_ICCVW21.pdf" target="_blank">Paper&nbsp</a>
<a href="https://drive.google.com/drive/folders/1BkWx0j6Kt5G8CEMzzREprMeoYfw0v4ge?usp=drive_link" target="_blank"><b>Dataset&nbsp</b></a>
</h4>


 SkyNet imrpoves sky-image prediction to model cloud dynamics with higher spatial and temporal resolution than previous works. Our method handles distorted clouds near the horizon of the hemispherical mirror by patially warping the sky images during training to facilitate longer forecasting of cloud evolution. 

```shell
# To download dataset for train and test data:
pip install gdown
gdown --folder --id 1BkWx0j6Kt5G8CEMzzREprMeoYfw0v4ge
 ```

# Installation

```shell
# Installation using using anaconda package management 
conda env create -f environment.yml
conda activate SkyNet
pip install -r requirements.txt
```

```shell
# How to train the model with default parameters:
python train.py
```

```shell
# For info about command-line flags use
python train.py --help
```

```shell
# Running Tests (WORK-IN-PROGRESS)
python test.py
```

# Thanks
This project makes use of LiteFlowNet for optical-flow estimates:
* [LiteFlowNet2](https://github.com/twhui/LiteFlowNet2) for lightweight optical-flow estimates using a CNN
Please refer to their webpage for installation and implementation

# Citation
If you use this project in your research please cite:
```
@inproceedings{ICCV_2021,
         author = {Leron Julian},
         title = {Precise Forecasting of Sky Images Using Spatial Warping},
         booktitle = {In Proceedings of the IEEE/CVF International Conference on Computer Vision},
         year = {2021}
     }
```



